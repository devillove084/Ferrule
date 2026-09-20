//! Deterministic, CPU-only distributed transaction protocol with explicit completion delivery.
use std::collections::{BTreeMap, BTreeSet};

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{ParallelRankId, ParticipantSet, ValidatedParallelTopology};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalizeOutcome {
    Success,
    Failure,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Commit,
    Abort,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionState {
    Preparing,
    Decided(Decision),
    Cancelled,
    Published,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransactionReport {
    pub rank: ParallelRankId,
    pub outcome: FinalizeOutcome,
}
/// The entire identity must match, not just the operation sequence number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CommunicationOperation {
    pub transaction: ExecutionTransactionId,
    pub rank: ParallelRankId,
    pub identity: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingCompletion {
    pub operation: CommunicationOperation,
    pub outcome: Option<FinalizeOutcome>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributedTransactionError {
    InvalidTransaction,
    InvalidRank,
    InvalidScope,
    InvalidState,
    /// A finalize ACK is only valid after a decision has been recorded.
    FinalizeBeforeDecision,
    /// A rank already submitted an immutable prepare vote or ACK.
    DuplicateReport,
    Backpressure,
    /// Retained transaction metadata is full; retire a terminal, drained transaction.
    TransactionLimitReached,
    /// Retained operation metadata is full, including records already drained.
    OperationLimitReached,
    StaleOperation,
    DuplicateOperation,
    IdentityExhausted,
}
/// Coordinator-wide retained metadata bounds, independent of in-flight credits.
/// Terminal transactions and drained operations count until explicit retirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistributedTransactionLimits {
    /// Includes terminal records until retirement. Zero disables transaction admission.
    pub max_transactions: usize,
    /// Includes drained records until retirement. Zero disables communication admission.
    pub max_operations: usize,
}
impl Default for DistributedTransactionLimits {
    /// Retain at most 1,024 transactions and 65,536 operations.
    fn default() -> Self {
        Self {
            max_transactions: 1_024,
            max_operations: 65_536,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct CommunicationRecord {
    outcome: Option<FinalizeOutcome>,
    drained: bool,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistributedTransactionRecord {
    id: ExecutionTransactionId,
    state: TransactionState,
    participants: ParticipantSet,
    /// Resource admission only, not a readiness vote or post-decision ACK.
    prepared: BTreeSet<ParallelRankId>,
    /// Explicit readiness results, including failures, are immutable decision evidence.
    prepare_votes: BTreeMap<ParallelRankId, FinalizeOutcome>,
    /// Post-decision rollback/quiescence or commit ACKs.
    finalize_acks: BTreeMap<ParallelRankId, FinalizeOutcome>,
    // Explicit reservations exclude credits owned by undrained operations.
    credits: usize,
    operations: BTreeMap<CommunicationOperation, CommunicationRecord>,
}
impl DistributedTransactionRecord {
    fn finalization_complete(&self) -> bool {
        self.operations.values().all(|record| record.drained)
            && self
                .participants
                .iter()
                .all(|rank| self.finalize_acks.get(&rank) == Some(&FinalizeOutcome::Success))
    }
}
#[derive(Debug, Clone)]
pub struct DistributedTransaction {
    topology: ValidatedParallelTopology,
    credit_capacity: usize,
    limits: DistributedTransactionLimits,
    retained_operations: usize,
    transaction_high_water: Option<ExecutionTransactionId>,
    in_use: usize,
    publication_count: usize,
    next_operation: u64,
    // Retirement releases records; a constant-space high-water mark prevents ID reuse.
    transactions: BTreeMap<ExecutionTransactionId, DistributedTransactionRecord>,
}
impl DistributedTransaction {
    /// Use default metadata limits: 1,024 transactions and 65,536 retained operations.
    /// Call `retire` on terminal, drained transactions to reclaim metadata capacity.
    pub fn new(topology: ValidatedParallelTopology, credit_capacity: usize) -> Self {
        Self::new_with_limits(
            topology,
            credit_capacity,
            DistributedTransactionLimits::default(),
        )
    }
    /// Set independent credit and metadata bounds. No records are evicted implicitly;
    /// reaching a metadata limit requires explicit retirement before admission can resume.
    pub fn new_with_limits(
        topology: ValidatedParallelTopology,
        credit_capacity: usize,
        limits: DistributedTransactionLimits,
    ) -> Self {
        Self {
            topology,
            credit_capacity,
            limits,
            retained_operations: 0,
            transaction_high_water: None,
            in_use: 0,
            publication_count: 0,
            next_operation: 1,
            transactions: BTreeMap::new(),
        }
    }
    pub const fn topology(&self) -> &ValidatedParallelTopology {
        &self.topology
    }
    /// Begin with all ranks. Successfully admitted IDs must be globally strictly increasing.
    /// A failed admission never advances the transaction high-water mark.
    pub fn begin(&mut self, id: ExecutionTransactionId) -> Result<(), DistributedTransactionError> {
        // Check capacity before allocating the default participant set.
        self.check_begin(id)?;
        self.begin_scoped(id, self.topology.participants())
    }
    fn check_begin(&self, id: ExecutionTransactionId) -> Result<(), DistributedTransactionError> {
        if self.transaction_high_water.is_some_and(|last| id <= last) {
            return Err(DistributedTransactionError::InvalidTransaction);
        }
        if self.transactions.len() >= self.limits.max_transactions {
            return Err(DistributedTransactionError::TransactionLimitReached);
        }
        Ok(())
    }
    /// Begin using the existing protocol, restricted to this transaction's scope.
    /// IDs must increase across all DP scopes, even after retirement. Failed begins
    /// consume no ID; completion and retirement order remain unrestricted.
    pub fn begin_scoped(
        &mut self,
        id: ExecutionTransactionId,
        participants: ParticipantSet,
    ) -> Result<(), DistributedTransactionError> {
        self.check_begin(id)?;
        self.topology
            .validate_participants(&participants)
            .map_err(|_| DistributedTransactionError::InvalidScope)?;
        self.transactions.insert(
            id,
            DistributedTransactionRecord {
                id,
                state: TransactionState::Preparing,
                participants,
                prepared: BTreeSet::new(),
                prepare_votes: BTreeMap::new(),
                finalize_acks: BTreeMap::new(),
                credits: 0,
                operations: BTreeMap::new(),
            },
        );
        self.transaction_high_water = Some(id);
        Ok(())
    }
    fn tx_mut(
        &mut self,
        id: ExecutionTransactionId,
    ) -> Result<&mut DistributedTransactionRecord, DistributedTransactionError> {
        self.transactions
            .get_mut(&id)
            .ok_or(DistributedTransactionError::InvalidTransaction)
    }
    /// Record resource admission only. The caller must separately submit a
    /// `prepare_vote` after compute/apply completes; admission is neither a
    /// readiness vote nor a post-decision ACK.
    pub fn prepare(
        &mut self,
        id: ExecutionTransactionId,
        rank: ParallelRankId,
    ) -> Result<(), DistributedTransactionError> {
        if rank.get() >= self.topology.world_size() {
            return Err(DistributedTransactionError::InvalidRank);
        }
        let tx = self.tx_mut(id)?;
        if !tx.participants.contains(rank) {
            return Err(DistributedTransactionError::InvalidRank);
        }
        if tx.state != TransactionState::Preparing {
            return Err(DistributedTransactionError::InvalidState);
        }
        tx.prepared
            .insert(rank)
            .then_some(())
            .ok_or(DistributedTransactionError::InvalidState)
    }

    /// Record an immutable pre-decision readiness result, never a finalize ACK.
    /// Success also prepares the rank. Failure can be reported before resource
    /// preparation succeeds, or after `prepare` if later pre-decision work fails.
    /// Once all participants have voted and communication drains, any failure
    /// forces Abort. Commit requires every participant's successful ready vote.
    pub fn prepare_vote(
        &mut self,
        id: ExecutionTransactionId,
        rank: ParallelRankId,
        outcome: FinalizeOutcome,
    ) -> Result<(), DistributedTransactionError> {
        if rank.get() >= self.topology.world_size() {
            return Err(DistributedTransactionError::InvalidRank);
        }
        let tx = self.tx_mut(id)?;
        if !tx.participants.contains(rank) {
            return Err(DistributedTransactionError::InvalidRank);
        }
        if tx.state != TransactionState::Preparing {
            return Err(DistributedTransactionError::InvalidState);
        }
        if tx.prepare_votes.contains_key(&rank) {
            return Err(DistributedTransactionError::DuplicateReport);
        }
        tx.prepare_votes.insert(rank, outcome);
        if outcome == FinalizeOutcome::Success {
            tx.prepared.insert(rank);
        }
        Ok(())
    }

    pub fn reserve(
        &mut self,
        id: ExecutionTransactionId,
        credits: usize,
    ) -> Result<(), DistributedTransactionError> {
        if self.tx_mut(id)?.state != TransactionState::Preparing {
            return Err(DistributedTransactionError::InvalidState);
        }
        if credits > self.credit_capacity - self.in_use {
            return Err(DistributedTransactionError::Backpressure);
        }
        self.tx_mut(id)?.credits += credits;
        self.in_use += credits;
        Ok(())
    }
    pub fn release(
        &mut self,
        id: ExecutionTransactionId,
        credits: usize,
    ) -> Result<(), DistributedTransactionError> {
        let tx = self.tx_mut(id)?;
        if credits > tx.credits {
            return Err(DistributedTransactionError::InvalidState);
        }
        tx.credits -= credits;
        self.in_use -= credits;
        Ok(())
    }
    /// Submit one rank's communication, owning one credit until completion is drained.
    pub fn communicate(
        &mut self,
        id: ExecutionTransactionId,
        rank: ParallelRankId,
    ) -> Result<CommunicationOperation, DistributedTransactionError> {
        if rank.get() >= self.topology.world_size() {
            return Err(DistributedTransactionError::InvalidRank);
        }
        let tx = self.tx_mut(id)?;
        if !tx.participants.contains(rank) {
            return Err(DistributedTransactionError::InvalidRank);
        }
        if tx.state != TransactionState::Preparing {
            return Err(DistributedTransactionError::InvalidState);
        }
        if self.in_use == self.credit_capacity {
            return Err(DistributedTransactionError::Backpressure);
        }
        if self.retained_operations >= self.limits.max_operations {
            return Err(DistributedTransactionError::OperationLimitReached);
        }
        let next = self
            .next_operation
            .checked_add(1)
            .ok_or(DistributedTransactionError::IdentityExhausted)?;
        let operation = CommunicationOperation {
            transaction: id,
            rank,
            identity: self.next_operation,
        };
        self.tx_mut(id)?.operations.insert(
            operation,
            CommunicationRecord {
                outcome: None,
                drained: false,
            },
        );
        self.next_operation = next;
        self.in_use += 1;
        self.retained_operations += 1;
        Ok(operation)
    }
    /// Atomically admit one operation per scoped participant, in rank order.
    /// Credit, metadata limit and identity failures leave all admission state unchanged.
    pub fn communicate_all(
        &mut self,
        id: ExecutionTransactionId,
    ) -> Result<Vec<CommunicationOperation>, DistributedTransactionError> {
        let tx = self
            .transactions
            .get(&id)
            .ok_or(DistributedTransactionError::InvalidTransaction)?;
        if tx.state != TransactionState::Preparing {
            return Err(DistributedTransactionError::InvalidState);
        }
        let count = tx.participants.len();
        if count > self.credit_capacity - self.in_use {
            return Err(DistributedTransactionError::Backpressure);
        }
        if count > self.limits.max_operations - self.retained_operations {
            return Err(DistributedTransactionError::OperationLimitReached);
        }
        let count_u64 =
            u64::try_from(count).map_err(|_| DistributedTransactionError::IdentityExhausted)?;
        let next = self
            .next_operation
            .checked_add(count_u64)
            .ok_or(DistributedTransactionError::IdentityExhausted)?;
        let operations: Vec<_> = tx
            .participants
            .iter()
            .zip(self.next_operation..next)
            .map(|(rank, identity)| CommunicationOperation {
                transaction: id,
                rank,
                identity,
            })
            .collect();
        let tx = self.tx_mut(id)?;
        for operation in &operations {
            tx.operations.insert(
                *operation,
                CommunicationRecord {
                    outcome: None,
                    drained: false,
                },
            );
        }
        self.next_operation = next;
        self.in_use += count;
        self.retained_operations += count;
        Ok(operations)
    }
    /// Record arrival without releasing its credit. Cancelled work still needs draining.
    pub fn complete(
        &mut self,
        operation: CommunicationOperation,
        outcome: FinalizeOutcome,
    ) -> Result<(), DistributedTransactionError> {
        let record = self
            .transactions
            .get_mut(&operation.transaction)
            .and_then(|tx| tx.operations.get_mut(&operation))
            .ok_or(DistributedTransactionError::StaleOperation)?;
        if record.outcome.is_some() {
            return Err(DistributedTransactionError::DuplicateOperation);
        }
        record.outcome = Some(outcome);
        Ok(())
    }
    /// Includes arrived completions until `drain` consumes them.
    pub fn pending_completions(&self) -> Vec<PendingCompletion> {
        self.transactions
            .values()
            .flat_map(|tx| {
                tx.operations.iter().filter_map(|(operation, record)| {
                    (!record.drained).then_some(PendingCompletion {
                        operation: *operation,
                        outcome: record.outcome,
                    })
                })
            })
            .collect()
    }
    /// Drain only arrived completions, retaining identities to reject replay.
    pub fn drain(&mut self) -> usize {
        let mut drained = 0;
        for tx in self.transactions.values_mut() {
            for record in tx.operations.values_mut() {
                if !record.drained && record.outcome.is_some() {
                    record.drained = true;
                    drained += 1;
                }
            }
        }
        self.in_use -= drained;
        drained
    }
    pub fn commit_decision(
        &mut self,
        id: ExecutionTransactionId,
    ) -> Result<Decision, DistributedTransactionError> {
        let tx = self.tx_mut(id)?;
        if tx.state != TransactionState::Preparing
            || tx
                .participants
                .iter()
                .any(|rank| !tx.prepare_votes.contains_key(&rank))
            || tx.operations.values().any(|record| !record.drained)
        {
            return Err(DistributedTransactionError::InvalidState);
        }
        let decision = if tx
            .prepare_votes
            .values()
            .any(|outcome| *outcome == FinalizeOutcome::Failure)
            || tx
                .operations
                .values()
                .any(|record| record.outcome == Some(FinalizeOutcome::Failure))
        {
            Decision::Abort
        } else if tx
            .participants
            .iter()
            .all(|rank| tx.prepare_votes.get(&rank) == Some(&FinalizeOutcome::Success))
        {
            Decision::Commit
        } else {
            return Err(DistributedTransactionError::InvalidState);
        };
        tx.state = TransactionState::Decided(decision);
        Ok(decision)
    }
    /// Record one post-decision commit or rollback/quiescence ACK.
    ///
    /// A pre-decision call is rejected instead of being treated as a vote. A
    /// failed Commit ACK leaves the transaction in Decided(Commit) and may be
    /// retried with a later successful ACK; it never changes the decision.
    /// Missing ACKs are Pending. Repeated Failure is idempotent for either
    /// decision; Success is immutable, including against a late Failure.
    pub fn finalize(
        &mut self,
        id: ExecutionTransactionId,
        rank: ParallelRankId,
        outcome: FinalizeOutcome,
    ) -> Result<(), DistributedTransactionError> {
        if rank.get() >= self.topology.world_size() {
            return Err(DistributedTransactionError::InvalidRank);
        }
        let tx = self.tx_mut(id)?;
        if !tx.participants.contains(rank) {
            return Err(DistributedTransactionError::InvalidRank);
        }
        if tx.state == TransactionState::Preparing {
            return Err(DistributedTransactionError::FinalizeBeforeDecision);
        }
        if !matches!(tx.state, TransactionState::Decided(_)) {
            return Err(DistributedTransactionError::InvalidState);
        }
        if tx.finalize_acks.get(&rank) == Some(&FinalizeOutcome::Success) {
            return Err(DistributedTransactionError::DuplicateReport);
        }
        tx.finalize_acks.insert(rank, outcome);
        Ok(())
    }

    /// Record an explicit Abort when cancellation or admission failure prevents
    /// collecting all ready votes. This does not prove cleanup or permit retirement.
    pub fn abort_decision(
        &mut self,
        id: ExecutionTransactionId,
    ) -> Result<Decision, DistributedTransactionError> {
        let record = self.tx_mut(id)?;
        if record.state != TransactionState::Preparing {
            return Err(DistributedTransactionError::InvalidState);
        }
        record.state = TransactionState::Decided(Decision::Abort);
        Ok(Decision::Abort)
    }
    /// Request cancellation, then poll it after rollback/quiescence ACKs.
    /// Success acknowledges the request, not cleanup: until all ACKs and drain
    /// complete, the state stays Decided(Abort) and reservations stay owned.
    pub fn cancel(
        &mut self,
        id: ExecutionTransactionId,
    ) -> Result<(), DistributedTransactionError> {
        let tx = self.tx_mut(id)?;
        if tx.state == TransactionState::Preparing {
            tx.state = TransactionState::Decided(Decision::Abort);
        } else if tx.state != TransactionState::Decided(Decision::Abort) {
            return Err(DistributedTransactionError::InvalidState);
        }
        if !tx.finalization_complete() {
            return Ok(());
        }
        tx.state = TransactionState::Cancelled;
        let credits = tx.credits;
        tx.credits = 0;
        self.in_use -= credits;
        Ok(())
    }
    /// Publish the decision once. Both Commit and Abort require every
    /// participant's post-decision Success ACK. Abort ACKs represent completed
    /// rollback/quiescence cleanup, not the pre-decision failure vote.
    pub fn publish(
        &mut self,
        id: ExecutionTransactionId,
    ) -> Result<(), DistributedTransactionError> {
        let tx = self.tx_mut(id)?;
        let ready = matches!(tx.state, TransactionState::Decided(_)) && tx.finalization_complete();
        if !ready {
            return Err(DistributedTransactionError::InvalidState);
        }
        tx.state = TransactionState::Published;
        let credits = tx.credits;
        tx.credits = 0;
        self.in_use -= credits;
        self.publication_count += 1;
        Ok(())
    }
    /// Remove a Cancelled or Published transaction only after every operation is drained.
    /// Draining alone never discards failure votes needed by an undecided transaction.
    /// Retired IDs stay invalid via the high-water mark, and their operations become stale.
    pub fn retire(
        &mut self,
        id: ExecutionTransactionId,
    ) -> Result<(), DistributedTransactionError> {
        let tx = self
            .transactions
            .get(&id)
            .ok_or(DistributedTransactionError::InvalidTransaction)?;
        if !matches!(
            tx.state,
            TransactionState::Cancelled | TransactionState::Published
        ) || !tx.finalization_complete()
        {
            return Err(DistributedTransactionError::InvalidState);
        }
        let count = tx.operations.len();
        self.transactions.remove(&id);
        self.retained_operations -= count;
        Ok(())
    }
    pub const fn limits(&self) -> DistributedTransactionLimits {
        self.limits
    }
    pub fn retained_transaction_count(&self) -> usize {
        self.transactions.len()
    }
    /// Includes completed and drained operations until their transaction is retired.
    pub const fn retained_operation_count(&self) -> usize {
        self.retained_operations
    }
    pub fn state(
        &self,
        id: ExecutionTransactionId,
    ) -> Result<TransactionState, DistributedTransactionError> {
        self.transactions
            .get(&id)
            .map(|t| t.state)
            .ok_or(DistributedTransactionError::InvalidTransaction)
    }
    pub fn participants(
        &self,
        id: ExecutionTransactionId,
    ) -> Result<&ParticipantSet, DistributedTransactionError> {
        self.transactions
            .get(&id)
            .map(|tx| &tx.participants)
            .ok_or(DistributedTransactionError::InvalidTransaction)
    }
    /// Participants without a successful post-decision ACK. All participants
    /// remain pending before the decision, regardless of resource-ready votes.
    pub fn pending_ranks(
        &self,
        id: ExecutionTransactionId,
    ) -> Result<Vec<ParallelRankId>, DistributedTransactionError> {
        let tx = self
            .transactions
            .get(&id)
            .ok_or(DistributedTransactionError::InvalidTransaction)?;
        Ok(tx
            .participants
            .iter()
            .filter(|rank| tx.finalize_acks.get(rank) != Some(&FinalizeOutcome::Success))
            .collect())
    }
    pub const fn publication_count(&self) -> usize {
        self.publication_count
    }
    pub const fn in_use_credits(&self) -> usize {
        self.in_use
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrule_common::{ParallelTopologyId, ParallelismPlan};

    #[test]
    fn batch_identity_exhaustion_is_atomic_and_preserves_single_operation_boundary() {
        let topology = ValidatedParallelTopology::new(
            ParallelTopologyId::new(1),
            4,
            ParallelRankId::new(0),
            ParallelismPlan::validated(2, 2, 1, 1, 1, 1).unwrap(),
        )
        .unwrap();
        for next in [u64::MAX - 1, u64::MAX] {
            let mut s = DistributedTransaction::new(topology.clone(), 8);
            let tx = ExecutionTransactionId::new(1).unwrap();
            let other = ExecutionTransactionId::new(2).unwrap();
            s.begin_scoped(tx, topology.tensor_participants(0).unwrap())
                .unwrap();
            s.begin_scoped(other, topology.tensor_participants(1).unwrap())
                .unwrap();
            s.reserve(tx, 1).unwrap();
            s.communicate(other, ParallelRankId::new(2)).unwrap();
            s.next_operation = next;
            let before = s.transactions.clone();
            let metadata = (s.retained_operations, s.transaction_high_water);
            for _ in 0..2 {
                assert_eq!(
                    s.communicate_all(tx),
                    Err(DistributedTransactionError::IdentityExhausted)
                );
                assert_eq!(s.transactions, before);
                assert_eq!((s.retained_operations, s.transaction_high_water), metadata);
                assert_eq!(s.in_use_credits(), 2);
                assert_eq!(s.next_operation, next);
                assert_eq!(s.publication_count(), 0);
            }
            if next == u64::MAX - 1 {
                let operation = s.communicate(tx, ParallelRankId::new(0)).unwrap();
                assert_eq!(operation.identity, next);
                assert_eq!(s.next_operation, u64::MAX);
            }
            let before = s.transactions.clone();
            let metadata = (s.retained_operations, s.transaction_high_water);
            let credits = s.in_use_credits();
            assert_eq!(
                s.communicate(tx, ParallelRankId::new(1)),
                Err(DistributedTransactionError::IdentityExhausted)
            );
            assert_eq!(s.transactions, before);
            assert_eq!((s.retained_operations, s.transaction_high_water), metadata);
            assert_eq!(s.in_use_credits(), credits);
        }

        let mut s = DistributedTransaction::new(topology.clone(), 2);
        let tx = ExecutionTransactionId::new(3).unwrap();
        s.begin_scoped(tx, topology.tensor_participants(1).unwrap())
            .unwrap();
        s.next_operation = u64::MAX - 2;
        let operations = s.communicate_all(tx).unwrap();
        assert_eq!(
            operations.iter().map(|op| op.identity).collect::<Vec<_>>(),
            vec![u64::MAX - 2, u64::MAX - 1]
        );
        assert_eq!(s.next_operation, u64::MAX);
        for operation in operations {
            s.complete(operation, FinalizeOutcome::Success).unwrap();
        }
        assert_eq!(s.drain(), 2);
        let before = s.transactions.clone();
        assert_eq!(
            s.communicate_all(tx),
            Err(DistributedTransactionError::IdentityExhausted)
        );
        assert_eq!(s.transactions, before);
        assert_eq!(s.retained_operation_count(), 2);
        assert_eq!(s.in_use_credits(), 0);
    }
}
