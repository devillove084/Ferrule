//! Deterministic, CPU-only distributed transaction protocol with explicit completion delivery.
use std::collections::{BTreeMap, BTreeSet};

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{ParallelRankId, ValidatedParallelTopology};

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
    InvalidState,
    DuplicateReport,
    Backpressure,
    StaleOperation,
    DuplicateOperation,
    IdentityExhausted,
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
    prepared: BTreeSet<ParallelRankId>,
    reports: BTreeMap<ParallelRankId, FinalizeOutcome>,
    // Explicit reservations exclude credits owned by undrained operations.
    credits: usize,
    operations: BTreeMap<CommunicationOperation, CommunicationRecord>,
}
#[derive(Debug, Clone)]
pub struct DistributedTransaction {
    topology: ValidatedParallelTopology,
    credit_capacity: usize,
    in_use: usize,
    publication_count: usize,
    next_operation: u64,
    // Retain terminal transactions so their identities cannot be reused.
    transactions: BTreeMap<ExecutionTransactionId, DistributedTransactionRecord>,
}
impl DistributedTransaction {
    pub fn new(topology: ValidatedParallelTopology, credit_capacity: usize) -> Self {
        Self {
            topology,
            credit_capacity,
            in_use: 0,
            publication_count: 0,
            next_operation: 1,
            transactions: BTreeMap::new(),
        }
    }
    pub const fn topology(&self) -> &ValidatedParallelTopology {
        &self.topology
    }
    pub fn begin(&mut self, id: ExecutionTransactionId) -> Result<(), DistributedTransactionError> {
        if self.transactions.contains_key(&id) {
            return Err(DistributedTransactionError::InvalidTransaction);
        }
        self.transactions.insert(
            id,
            DistributedTransactionRecord {
                id,
                state: TransactionState::Preparing,
                prepared: BTreeSet::new(),
                reports: BTreeMap::new(),
                credits: 0,
                operations: BTreeMap::new(),
            },
        );
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
    pub fn prepare(
        &mut self,
        id: ExecutionTransactionId,
        rank: ParallelRankId,
    ) -> Result<(), DistributedTransactionError> {
        if !self.topology.participants().contains(rank) {
            return Err(DistributedTransactionError::InvalidRank);
        }
        let tx = self.tx_mut(id)?;
        if tx.state != TransactionState::Preparing || !tx.prepared.insert(rank) {
            return Err(DistributedTransactionError::InvalidState);
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
        if !self.topology.participants().contains(rank) {
            return Err(DistributedTransactionError::InvalidRank);
        }
        if self.tx_mut(id)?.state != TransactionState::Preparing {
            return Err(DistributedTransactionError::InvalidState);
        }
        if self.in_use == self.credit_capacity {
            return Err(DistributedTransactionError::Backpressure);
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
        Ok(operation)
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
        let participants = self.topology.participants();
        let tx = self.tx_mut(id)?;
        if tx.state != TransactionState::Preparing
            || participants.iter().any(|rank| !tx.prepared.contains(&rank))
            || tx.operations.values().any(|record| !record.drained)
        {
            return Err(DistributedTransactionError::InvalidState);
        }
        let decision = if tx.reports.values().any(|o| *o == FinalizeOutcome::Failure)
            || tx
                .operations
                .values()
                .any(|record| record.outcome == Some(FinalizeOutcome::Failure))
        {
            Decision::Abort
        } else {
            Decision::Commit
        };
        tx.state = TransactionState::Decided(decision);
        Ok(decision)
    }
    /// Before a decision, reports are immutable votes and failure forces Abort.
    /// After Commit, Failure records a retryable attempt, not successful finalization:
    /// the rank stays pending until Success. Success can never be overwritten.
    pub fn finalize(
        &mut self,
        id: ExecutionTransactionId,
        rank: ParallelRankId,
        outcome: FinalizeOutcome,
    ) -> Result<(), DistributedTransactionError> {
        if !self.topology.participants().contains(rank) {
            return Err(DistributedTransactionError::InvalidRank);
        }
        let tx = self.tx_mut(id)?;
        if !matches!(
            tx.state,
            TransactionState::Preparing | TransactionState::Decided(_)
        ) {
            return Err(DistributedTransactionError::InvalidState);
        }
        if let Some(previous) = tx.reports.get(&rank) {
            if tx.state != TransactionState::Decided(Decision::Commit)
                || *previous == FinalizeOutcome::Success
            {
                return Err(DistributedTransactionError::DuplicateReport);
            }
        }
        tx.reports.insert(rank, outcome);
        Ok(())
    }
    pub fn cancel(
        &mut self,
        id: ExecutionTransactionId,
    ) -> Result<(), DistributedTransactionError> {
        let credits = {
            let tx = self.tx_mut(id)?;
            if !matches!(
                tx.state,
                TransactionState::Preparing | TransactionState::Decided(Decision::Abort)
            ) {
                return Err(DistributedTransactionError::InvalidState);
            }
            let credits = tx.credits;
            tx.credits = 0;
            tx.state = TransactionState::Cancelled;
            credits
        };
        // Operation credits stay owned until a late completion is drained.
        self.in_use -= credits;
        Ok(())
    }
    /// Publish the decision once. Commit requires every participant's Success;
    /// Abort only requires every participant to have reported.
    pub fn publish(
        &mut self,
        id: ExecutionTransactionId,
    ) -> Result<(), DistributedTransactionError> {
        let participants = self.topology.participants();
        let tx = self.tx_mut(id)?;
        let ready = match tx.state {
            TransactionState::Decided(Decision::Commit) => participants
                .iter()
                .all(|rank| tx.reports.get(&rank) == Some(&FinalizeOutcome::Success)),
            TransactionState::Decided(Decision::Abort) => participants
                .iter()
                .all(|rank| tx.reports.contains_key(&rank)),
            _ => false,
        };
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
    pub fn state(
        &self,
        id: ExecutionTransactionId,
    ) -> Result<TransactionState, DistributedTransactionError> {
        self.transactions
            .get(&id)
            .map(|t| t.state)
            .ok_or(DistributedTransactionError::InvalidTransaction)
    }
    pub fn pending_ranks(
        &self,
        id: ExecutionTransactionId,
    ) -> Result<Vec<ParallelRankId>, DistributedTransactionError> {
        let tx = self
            .transactions
            .get(&id)
            .ok_or(DistributedTransactionError::InvalidTransaction)?;
        Ok(self
            .topology
            .participants()
            .iter()
            .filter(|rank| match tx.state {
                TransactionState::Decided(Decision::Commit) => {
                    tx.reports.get(rank) != Some(&FinalizeOutcome::Success)
                }
                _ => !tx.reports.contains_key(rank),
            })
            .collect())
    }
    pub const fn publication_count(&self) -> usize {
        self.publication_count
    }
    pub const fn in_use_credits(&self) -> usize {
        self.in_use
    }
}
