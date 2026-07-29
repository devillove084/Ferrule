//! Generic native multi-session executor.
//!
//! This executor wraps any [`MultiSessionRunner`] and implements the neutral
//! `ExecutionBatch` -> `ExecutionOutput` contract for batches containing
//! multiple sequences, ragged prefill chunks, and mixed prefill/decode rows.

use std::collections::HashMap;

use ferrule_common::execution::{
    ExecutionBatch, ExecutionCapabilities, ExecutionOutput, ExecutionTransactionId,
    KvReservationView,
};
use ferrule_common::{ContinuationId, Error, ExpertLeaseSet, Result};
use ferrule_model::{
    BatchContinuationCancelOutcome, MultiSessionBatchProgress, MultiSessionRunner,
    PendingModelProgress,
};

use crate::expert_residency::ExpertResidencyController;
use crate::scheduling::{ExpertIoResourceBrokerHandle, ResourceBrokerStats, ResourceSnapshot};

/// Progress from a resumable native multi-session batch.
#[derive(Debug, Clone, PartialEq)]
pub enum NativeBatchExecutionProgress {
    Complete(ExecutionOutput),
    Waiting(PendingModelProgress),
}

/// Native multi-session executor wrapping any [`MultiSessionRunner`].
///
/// Every prepared or suspended execution is tracked by its stable transaction
/// identity. Transactions may be prepared, waiting, resumed, and finalized in
/// any order; continuation ownership remains local to the transaction that
/// created it.
pub struct NativeMultiSessionExecutor<R: MultiSessionRunner> {
    runner: R,
    capabilities: ExecutionCapabilities,
    poison: Option<PoisonState>,
    transactions: HashMap<ExecutionTransactionId, NativeTransaction>,
    expert_residency_initialized: bool,
    expert_io_resources: Option<ExpertIoResourceBrokerHandle>,
}

#[derive(Debug)]
struct NativeTransaction {
    batch: ExecutionBatch,
    pending_model_progress: Option<PendingModelProgress>,
}

#[derive(Debug)]
struct PoisonState {
    operation: &'static str,
    cause: String,
}

impl<R: MultiSessionRunner> NativeMultiSessionExecutor<R> {
    /// Wrap a runner with native multi-session capabilities.
    pub fn new(runner: R) -> Self {
        let capabilities = runner.multi_session_capabilities();
        Self {
            runner,
            capabilities,
            poison: None,
            transactions: HashMap::new(),
            expert_residency_initialized: false,
            expert_io_resources: None,
        }
    }

    pub(crate) fn with_expert_io_resources(
        mut self,
        resources: ExpertIoResourceBrokerHandle,
    ) -> Self {
        self.expert_io_resources = Some(resources);
        self
    }

    pub fn expert_io_resource_snapshots(&self) -> Result<Vec<ResourceSnapshot>> {
        self.expert_io_resources
            .as_ref()
            .map_or_else(|| Ok(Vec::new()), ExpertIoResourceBrokerHandle::snapshots)
    }

    pub fn expert_io_resource_stats(&self) -> Result<Option<ResourceBrokerStats>> {
        self.expert_io_resources
            .as_ref()
            .map(ExpertIoResourceBrokerHandle::stats)
            .transpose()
    }

    pub fn active_expert_io_grants(&self) -> Result<usize> {
        self.expert_io_resources
            .as_ref()
            .map_or_else(|| Ok(0), ExpertIoResourceBrokerHandle::active_grants)
    }

    /// Returns the truthful capabilities of the native multi-session path.
    pub fn capabilities(&self) -> &ExecutionCapabilities {
        &self.capabilities
    }

    /// Returns a reference to the underlying runner for runtime-owned policy and
    /// publication logic.
    pub(crate) fn runner(&self) -> &R {
        &self.runner
    }

    /// Returns a mutable reference to the underlying runner for runtime-owned
    /// lifecycle operations.
    pub(crate) fn runner_mut(&mut self) -> &mut R {
        &mut self.runner
    }

    /// Configure backend physical KV capacity and refresh truthful capabilities.
    pub fn configure_kv_page_capacity(&mut self, max_pages: usize) -> Result<()> {
        self.ensure_registry_empty("reconfigure KV capacity")?;
        self.runner.configure_kv_page_capacity(max_pages)?;
        self.capabilities = self.runner.multi_session_capabilities();
        Ok(())
    }

    /// Describe why the runner cannot currently be extracted without consuming
    /// the executor or abandoning retained transaction resources.
    pub(crate) fn runner_extraction_error(&self) -> Option<Error> {
        if !self.transactions.is_empty() {
            return Some(Error::Execution(format!(
                "cannot extract runner with active execution transactions {:?}",
                self.sorted_transaction_ids()
            )));
        }
        self.poison.as_ref().map(|poison| {
            Error::Execution(format!(
                "cannot extract runner after {} failed: {}",
                poison.operation, poison.cause
            ))
        })
    }

    /// Extract the runner or return the untouched executor with the error.
    pub fn try_into_runner(self) -> std::result::Result<R, Box<(Error, Self)>> {
        if let Some(error) = self.runner_extraction_error() {
            return Err(Box::new((error, self)));
        }
        Ok(self.runner)
    }

    /// Whether a shared model mutation failed and may have left state inconsistent.
    pub fn is_poisoned(&self) -> bool {
        self.poison.is_some()
    }

    pub fn poison_operation(&self) -> Option<&'static str> {
        self.poison.as_ref().map(|poison| poison.operation)
    }

    pub fn poison_cause(&self) -> Option<&str> {
        self.poison.as_ref().map(|poison| poison.cause.as_str())
    }

    /// Return the current model-owned wait state for one transaction.
    pub fn pending_model_progress(
        &self,
        transaction: ExecutionTransactionId,
    ) -> Option<&PendingModelProgress> {
        self.transactions
            .get(&transaction)
            .and_then(|transaction| transaction.pending_model_progress.as_ref())
    }

    /// Iterate over registered transaction IDs in arbitrary order.
    pub fn transaction_ids(&self) -> impl Iterator<Item = ExecutionTransactionId> + '_ {
        self.transactions.keys().copied()
    }

    /// Whether any prepared, completed, or suspended transaction is registered.
    pub fn has_transactions(&self) -> bool {
        !self.transactions.is_empty()
    }

    /// Whether one exact transaction identity remains registered.
    pub fn has_transaction(&self, transaction: ExecutionTransactionId) -> bool {
        self.transactions.contains_key(&transaction)
    }

    /// Reset the default runner session and clear poison.
    pub fn reset(&mut self) -> Result<()> {
        self.ensure_registry_empty("reset the native executor")?;
        match self.runner.reset_session() {
            Ok(()) => {
                self.poison = None;
                Ok(())
            }
            Err(error) => {
                self.record_poison("reset", &error);
                Err(error)
            }
        }
    }

    /// Mark the executor as poisoned after a runtime-side failure that may have
    /// left shared model state inconsistent.
    pub fn poison(&mut self, operation: &'static str, cause: &Error) {
        self.record_poison(operation, cause);
    }

    /// Construct a fresh independent sequence state at position zero.
    pub fn create_sequence_state(&mut self) -> Result<R::SequenceState> {
        self.ensure_ready()?;
        self.runner.create_sequence_state()
    }

    /// Prepare a new model state from one explicit committed source state.
    pub fn fork_sequence_state_from(
        &mut self,
        source: &R::SequenceState,
        expected_position: usize,
    ) -> Result<R::SequenceState> {
        self.ensure_ready()?;
        self.runner
            .fork_sequence_state_from(source, expected_position)
    }

    /// Reset a sequence state for reuse with a new logical sequence.
    pub fn reset_sequence_state(&mut self, state: &mut R::SequenceState) -> Result<()> {
        self.ensure_ready()?;
        self.runner.reset_sequence_state(state)
    }

    /// Release a sequence state and its physical capacity.
    pub fn release_sequence_state(&mut self, state: R::SequenceState) -> Result<()> {
        self.runner.release_sequence_state(state)
    }

    /// Execute one owner-thread operation against an explicit sequence state.
    pub fn with_sequence_state<T>(
        &mut self,
        state: &mut R::SequenceState,
        execute: impl FnOnce(&mut R) -> Result<T>,
    ) -> Result<T> {
        self.ensure_execution_ready()?;
        self.runner.with_sequence_state(state, execute)
    }

    /// Release physical pages whose runtime refcount reached zero.
    pub fn release_kv_pages(
        &mut self,
        pages: &[ferrule_common::execution::KvPageId],
    ) -> Result<()> {
        self.ensure_ready()?;
        self.runner.release_kv_pages(pages)
    }

    pub fn preempt_kv_pages(
        &mut self,
        pages: &[ferrule_common::execution::KvPageId],
    ) -> Result<()> {
        self.ensure_ready()?;
        self.ensure_registry_empty("preempt KV pages")?;
        self.runner.preempt_kv_pages(pages)
    }

    pub fn restore_kv_pages(
        &mut self,
        pages: &[ferrule_common::execution::KvPageId],
    ) -> Result<()> {
        self.ensure_ready()?;
        self.ensure_registry_empty("restore KV pages")?;
        self.runner.restore_kv_pages(pages)
    }

    /// Commit one completed backend transaction exactly once.
    pub fn commit_prepared_batch(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [R::SequenceState],
    ) -> Result<()> {
        self.quiescent_transaction(transaction)?;
        self.runner
            .commit_multi_session_batch(transaction, states)?;
        self.remove_transaction(transaction);
        Ok(())
    }

    /// Roll back one completed backend transaction exactly once.
    pub fn rollback_prepared_batch(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [R::SequenceState],
    ) -> Result<()> {
        self.quiescent_transaction(transaction)?;
        self.rollback_transaction_inner(transaction, states)
    }

    /// Retain exact model-owned prefixes for an entire provisional cohort while
    /// leaving one prepared backend KV reservation pending for commit.
    pub fn retain_prepared_prefixes(
        &mut self,
        transaction: ExecutionTransactionId,
        sources: &[R::SequenceState],
        branches: &mut [R::SequenceState],
        executed_rows: &[usize],
        retained_rows: &[usize],
    ) -> Result<()> {
        self.ensure_ready()?;
        self.quiescent_transaction(transaction)?;
        let sequence_count = sources.len();
        if sequence_count == 0
            || branches.len() != sequence_count
            || executed_rows.len() != sequence_count
            || retained_rows.len() != sequence_count
        {
            return Err(Error::Execution(format!(
                "provisional prefix cohort shape mismatch: sources={} branches={} executed={} retained={}",
                sequence_count,
                branches.len(),
                executed_rows.len(),
                retained_rows.len()
            )));
        }
        for (sequence, (&executed, &retained)) in
            executed_rows.iter().zip(retained_rows).enumerate()
        {
            if executed == 0 || retained == 0 || retained > executed {
                return Err(Error::Execution(format!(
                    "invalid provisional prefix for sequence {sequence}: retained={retained} executed={executed}"
                )));
            }
        }
        self.runner.retain_provisional_prefixes(
            transaction,
            sources,
            branches,
            executed_rows,
            retained_rows,
        )
    }

    /// Start a packed batch that may suspend on model-owned work.
    pub fn begin_resumable_batch_with_kv(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [R::SequenceState],
        batch: &ExecutionBatch,
        kv_reservations: &[KvReservationView],
    ) -> Result<NativeBatchExecutionProgress> {
        self.ensure_execution_ready()?;
        if self.transactions.contains_key(&transaction) {
            return Err(Error::Execution(format!(
                "execution transaction {transaction:?} is already registered"
            )));
        }
        batch.validate(states.len(), &self.capabilities)?;

        self.runner
            .prepare_multi_session_batch(transaction, states, batch, kv_reservations)?;
        self.transactions.insert(
            transaction,
            NativeTransaction {
                batch: batch.clone(),
                pending_model_progress: None,
            },
        );

        let progress =
            match self
                .runner
                .execute_multi_session_batch_progress(transaction, states, batch)
            {
                Ok(progress) => progress,
                Err(error) => {
                    return Err(self.fail_transaction(
                        transaction,
                        states,
                        "model execution",
                        error,
                    ));
                }
            };
        self.handle_progress(transaction, states, batch, None, progress)
    }

    /// Resume one suspended packed batch after its model-owned work is ready.
    pub fn resume_resumable_batch(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [R::SequenceState],
        batch: &ExecutionBatch,
        continuation: ContinuationId,
        leases: ExpertLeaseSet,
    ) -> Result<NativeBatchExecutionProgress> {
        self.ensure_ready()?;
        self.ensure_continuation_owner(transaction, continuation)?;
        let expected_batch = &self
            .transactions
            .get(&transaction)
            .expect("continuation index must reference a registered transaction")
            .batch;
        if expected_batch != batch {
            return Err(Error::Execution(format!(
                "resume batch does not exactly match transaction {transaction:?}"
            )));
        }
        batch.validate(states.len(), &self.capabilities)?;

        let progress = self.runner.resume_multi_session_batch(
            transaction,
            states,
            batch,
            continuation,
            leases,
        )?;
        self.handle_progress(transaction, states, batch, Some(continuation), progress)
    }

    /// Cancel one suspended batch and roll back only its backend transaction.
    pub fn cancel_resumable_batch(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [R::SequenceState],
        continuation: ContinuationId,
    ) -> Result<()> {
        self.ensure_continuation_owner(transaction, continuation)?;
        let model_cleanup_error = match self.runner.cancel_multi_session_batch(
            transaction,
            states,
            continuation,
        ) {
            BatchContinuationCancelOutcome::Cancelled => None,
            BatchContinuationCancelOutcome::StillActive(cancel_error) => {
                return Err(Error::Execution(format!(
                    "transaction {transaction:?} cancellation did not confirm quiescence: {cancel_error}"
                )));
            }
            BatchContinuationCancelOutcome::Quiesced(cleanup_error) => Some(cleanup_error),
        };

        self.clear_continuation(transaction, continuation);
        let rollback_error = self.rollback_transaction_inner(transaction, states).err();
        match (model_cleanup_error, rollback_error) {
            (None, None) => Ok(()),
            (None, Some(rollback_error)) => Err(Error::Execution(format!(
                "transaction {transaction:?} cancellation completed but backend rollback failed: {rollback_error}"
            ))),
            (Some(cleanup_error), None) => Err(Error::Execution(format!(
                "transaction {transaction:?} model work quiesced but cleanup failed: {cleanup_error}"
            ))),
            (Some(cleanup_error), Some(rollback_error)) => Err(Error::Execution(format!(
                "transaction {transaction:?} model work quiesced but cleanup failed ({cleanup_error}); backend rollback also failed ({rollback_error})"
            ))),
        }
    }

    fn handle_progress(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [R::SequenceState],
        batch: &ExecutionBatch,
        previous_continuation: Option<ContinuationId>,
        progress: MultiSessionBatchProgress,
    ) -> Result<NativeBatchExecutionProgress> {
        match progress {
            MultiSessionBatchProgress::Complete(output) => {
                if let Some(continuation) = previous_continuation {
                    self.clear_continuation(transaction, continuation);
                }
                match output.validate_with_capabilities(batch, &self.capabilities) {
                    Ok(()) => Ok(NativeBatchExecutionProgress::Complete(output)),
                    Err(error) => {
                        Err(self.fail_transaction(transaction, states, "output contract", error))
                    }
                }
            }
            MultiSessionBatchProgress::Waiting(pending) => {
                if pending.transaction() != transaction {
                    let error = Error::Execution(format!(
                        "runner returned waiting progress for transaction {:?} while executing transaction {transaction:?}",
                        pending.transaction()
                    ));
                    return Err(self.reject_invalid_waiting_progress(
                        transaction,
                        states,
                        pending,
                        error,
                    ));
                }
                self.transition_to_waiting(transaction, previous_continuation, &pending)?;
                Ok(NativeBatchExecutionProgress::Waiting(pending))
            }
        }
    }

    fn transition_to_waiting(
        &mut self,
        transaction: ExecutionTransactionId,
        previous_continuation: Option<ContinuationId>,
        pending: &PendingModelProgress,
    ) -> Result<()> {
        let current_continuation = self
            .transactions
            .get(&transaction)
            .ok_or_else(|| {
                Error::Execution(format!(
                    "waiting progress belongs to unregistered transaction {transaction:?}"
                ))
            })?
            .pending_model_progress
            .as_ref()
            .map(PendingModelProgress::continuation);
        if current_continuation != previous_continuation {
            let error = Error::Execution(format!(
                "transaction {transaction:?} expected previous continuation {current_continuation:?}, runner progress followed {previous_continuation:?}"
            ));
            self.record_poison("resumable_batch_contract", &error);
            return Err(error);
        }
        self.transactions
            .get_mut(&transaction)
            .expect("waiting progress transaction was validated above")
            .pending_model_progress = Some(pending.clone());
        Ok(())
    }

    fn reject_invalid_waiting_progress(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [R::SequenceState],
        pending: PendingModelProgress,
        contract_error: Error,
    ) -> Error {
        self.record_poison("resumable_batch_contract", &contract_error);
        let continuation = pending.continuation();
        match self
            .runner
            .cancel_multi_session_batch(transaction, states, continuation)
        {
            BatchContinuationCancelOutcome::Cancelled => self.fail_transaction(
                transaction,
                states,
                "waiting progress contract",
                contract_error,
            ),
            BatchContinuationCancelOutcome::Quiesced(cleanup_error) => {
                let combined = Error::Execution(format!(
                    "{contract_error}; model continuation quiesced with cleanup failure: {cleanup_error}"
                ));
                self.fail_transaction(transaction, states, "waiting progress contract", combined)
            }
            BatchContinuationCancelOutcome::StillActive(cancel_error) => {
                let corrected = PendingModelProgress::new(
                    transaction,
                    continuation,
                    pending.dependencies().clone(),
                )
                .expect("validated pending dependencies remain valid under corrected ownership");
                self.transactions
                    .get_mut(&transaction)
                    .expect("invalid progress must still belong to a registered transaction")
                    .pending_model_progress = Some(corrected);
                Error::Execution(format!(
                    "{contract_error}; cancellation did not quiesce the invalid continuation: {cancel_error}"
                ))
            }
        }
    }

    fn fail_transaction(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [R::SequenceState],
        failure_context: &'static str,
        error: Error,
    ) -> Error {
        match self.rollback_transaction_inner(transaction, states) {
            Ok(()) => error,
            Err(rollback_error) => Error::Execution(format!(
                "{failure_context} failed ({error}); backend rollback also failed ({rollback_error})"
            )),
        }
    }

    fn rollback_transaction_inner(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [R::SequenceState],
    ) -> Result<()> {
        if !self.transactions.contains_key(&transaction) {
            return Err(Error::Execution(format!(
                "execution transaction {transaction:?} is not registered"
            )));
        }
        if let Err(error) = self
            .runner
            .rollback_multi_session_batch(transaction, states)
        {
            self.record_poison("backend_kv_rollback", &error);
            return Err(error);
        }
        self.remove_transaction(transaction);
        Ok(())
    }

    fn remove_transaction(&mut self, transaction: ExecutionTransactionId) {
        self.transactions.remove(&transaction);
    }

    fn clear_continuation(
        &mut self,
        transaction: ExecutionTransactionId,
        continuation: ContinuationId,
    ) {
        if let Some(transaction_state) = self.transactions.get_mut(&transaction)
            && transaction_state
                .pending_model_progress
                .as_ref()
                .is_some_and(|pending| pending.continuation() == continuation)
        {
            transaction_state.pending_model_progress = None;
        }
    }

    fn ensure_continuation_owner(
        &self,
        transaction: ExecutionTransactionId,
        continuation: ContinuationId,
    ) -> Result<()> {
        let pending = self
            .transactions
            .get(&transaction)
            .and_then(|transaction| transaction.pending_model_progress.as_ref())
            .ok_or_else(|| {
                Error::Execution(format!(
                    "transaction {transaction:?} has no active model continuation"
                ))
            })?;
        if pending.continuation() != continuation {
            return Err(Error::Execution(format!(
                "transaction {transaction:?} owns continuation {:?}, not {continuation:?}",
                pending.continuation()
            )));
        }
        Ok(())
    }

    fn quiescent_transaction(
        &self,
        transaction: ExecutionTransactionId,
    ) -> Result<&NativeTransaction> {
        let transaction_state = self.transactions.get(&transaction).ok_or_else(|| {
            Error::Execution(format!(
                "execution transaction {transaction:?} is not registered"
            ))
        })?;
        if let Some(pending) = &transaction_state.pending_model_progress {
            return Err(Error::Execution(format!(
                "cannot finalize transaction {transaction:?} with active batch continuation {:?}; resume or cancel it first",
                pending.continuation()
            )));
        }
        Ok(transaction_state)
    }

    fn ensure_registry_empty(&self, operation: &str) -> Result<()> {
        if self.transactions.is_empty() {
            return Ok(());
        }
        Err(Error::Execution(format!(
            "cannot {operation} with active execution transactions {:?}",
            self.sorted_transaction_ids()
        )))
    }

    fn sorted_transaction_ids(&self) -> Vec<ExecutionTransactionId> {
        let mut transactions: Vec<_> = self.transactions.keys().copied().collect();
        transactions.sort_unstable();
        transactions
    }

    fn ensure_ready(&self) -> Result<()> {
        let Some(poison) = &self.poison else {
            return Ok(());
        };
        Err(Error::Execution(format!(
            "native executor is poisoned after {} failed: {}; reset before executing again",
            poison.operation, poison.cause
        )))
    }

    fn ensure_execution_ready(&mut self) -> Result<()> {
        self.ensure_ready()?;
        if self.expert_residency_initialized || self.runner.expert_residency_control_installed() {
            self.expert_residency_initialized = true;
            return Ok(());
        }

        let Some(requirements) = self.runner.expert_residency_requirements() else {
            self.expert_residency_initialized = true;
            return Ok(());
        };

        let result =
            ExpertResidencyController::with_requirements(requirements).and_then(|control| {
                self.runner
                    .install_expert_residency_control(Box::new(control))
            });
        match result {
            Ok(()) => {
                self.expert_residency_initialized = true;
                Ok(())
            }
            Err(error) => {
                self.record_poison("expert_residency_initialization", &error);
                Err(error)
            }
        }
    }

    fn record_poison(&mut self, operation: &'static str, error: &Error) {
        self.poison = Some(PoisonState {
            operation,
            cause: error.to_string(),
        });
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet, VecDeque};
    use std::num::NonZeroU32;

    use super::*;
    use ferrule_common::execution::{
        ExecutionSequence, ForwardMode, ForwardPhase, LogitsOutput, LogitsRequest, LogitsRow,
        StateSlot, TokenLogit,
    };
    use ferrule_common::{
        BackendId, DependencySet, DeviceId, DispatchFenceContract, FenceId, LogicalDependency,
        MappingEpoch, OperationId,
    };
    use ferrule_model::{ModelInfo, ModelRunner};

    #[derive(Debug, Default)]
    struct MockSequenceState {
        position: usize,
    }

    #[derive(Debug)]
    enum MockCancelOutcome {
        Cancelled,
        StillActive,
        Quiesced,
    }

    #[derive(Debug, Default)]
    struct MockMultiSessionRunner {
        progress: HashMap<ExecutionTransactionId, VecDeque<Result<MultiSessionBatchProgress>>>,
        cancel_outcomes: HashMap<ExecutionTransactionId, VecDeque<MockCancelOutcome>>,
        fail_prepares: HashSet<ExecutionTransactionId>,
        fail_rollbacks: HashSet<ExecutionTransactionId>,
        prepares: Vec<ExecutionTransactionId>,
        executes: Vec<ExecutionTransactionId>,
        resumes: Vec<(ExecutionTransactionId, ContinuationId)>,
        cancels: Vec<(ExecutionTransactionId, ContinuationId)>,
        commits: Vec<ExecutionTransactionId>,
        rollbacks: Vec<ExecutionTransactionId>,
        sequences_created: usize,
        sequences_forked: usize,
        sequences_released: usize,
    }

    impl MockMultiSessionRunner {
        fn push_progress(
            &mut self,
            transaction: ExecutionTransactionId,
            progress: Result<MultiSessionBatchProgress>,
        ) {
            self.progress
                .entry(transaction)
                .or_default()
                .push_back(progress);
        }

        fn push_cancel_outcome(
            &mut self,
            transaction: ExecutionTransactionId,
            outcome: MockCancelOutcome,
        ) {
            self.cancel_outcomes
                .entry(transaction)
                .or_default()
                .push_back(outcome);
        }

        fn next_progress(
            &mut self,
            transaction: ExecutionTransactionId,
        ) -> Result<MultiSessionBatchProgress> {
            self.progress
                .get_mut(&transaction)
                .and_then(VecDeque::pop_front)
                .unwrap_or_else(|| {
                    Err(Error::Execution(format!(
                        "mock has no progress for transaction {transaction:?}"
                    )))
                })
        }
    }

    impl ModelRunner for MockMultiSessionRunner {
        fn model_info(&self) -> ModelInfo {
            ModelInfo {
                family: ferrule_model::ModelFamily::Unknown("mock".into()),
                architecture: Some("mock".into()),
                attention: ferrule_model::AttentionKind::Unknown("mock".into()),
                weight_source: ferrule_model::WeightSource::Unknown,
                hidden_size: 4,
                num_layers: 1,
                num_experts: 0,
                num_experts_per_tok: 0,
                vocab_size: 8,
                backend: "mock",
            }
        }

        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text.bytes().map(u32::from).collect())
        }

        fn decode(&self, tokens: &[u32]) -> Result<String> {
            Ok(tokens
                .iter()
                .map(|token| char::from_u32(*token).unwrap_or('?'))
                .collect())
        }

        fn reset_session(&mut self) -> Result<()> {
            Ok(())
        }

        fn eos_token_id(&self) -> Option<u32> {
            None
        }
    }

    impl MultiSessionRunner for MockMultiSessionRunner {
        type SequenceState = MockSequenceState;

        fn sequence_generation(&self, _state: &Self::SequenceState) -> u64 {
            0
        }

        fn with_sequence_state<T>(
            &mut self,
            _state: &mut Self::SequenceState,
            execute: impl FnOnce(&mut Self) -> Result<T>,
        ) -> Result<T> {
            execute(self)
        }

        fn create_sequence_state(&mut self) -> Result<Self::SequenceState> {
            self.sequences_created += 1;
            Ok(MockSequenceState::default())
        }

        fn fork_sequence_state(&mut self) -> Result<Self::SequenceState> {
            self.sequences_forked += 1;
            Ok(MockSequenceState::default())
        }

        fn fork_sequence_state_from(
            &mut self,
            source: &Self::SequenceState,
            expected_position: usize,
        ) -> Result<Self::SequenceState> {
            if source.position != expected_position {
                return Err(Error::Execution("mock exact fork position mismatch".into()));
            }
            self.sequences_forked += 1;
            Ok(MockSequenceState {
                position: source.position,
            })
        }

        fn reset_sequence_state(&mut self, state: &mut Self::SequenceState) -> Result<()> {
            state.position = 0;
            Ok(())
        }

        fn release_sequence_state(&mut self, _state: Self::SequenceState) -> Result<()> {
            self.sequences_released += 1;
            Ok(())
        }

        fn configure_kv_page_capacity(&mut self, _max_pages: usize) -> Result<()> {
            Ok(())
        }

        fn release_kv_pages(
            &mut self,
            _pages: &[ferrule_common::execution::KvPageId],
        ) -> Result<()> {
            Ok(())
        }

        fn preempt_kv_pages(
            &mut self,
            _pages: &[ferrule_common::execution::KvPageId],
        ) -> Result<()> {
            Ok(())
        }

        fn restore_kv_pages(
            &mut self,
            _pages: &[ferrule_common::execution::KvPageId],
        ) -> Result<()> {
            Ok(())
        }

        fn prepare_multi_session_batch(
            &mut self,
            transaction: ExecutionTransactionId,
            _states: &mut [Self::SequenceState],
            _batch: &ExecutionBatch,
            _kv_reservations: &[KvReservationView],
        ) -> Result<()> {
            self.prepares.push(transaction);
            if self.fail_prepares.contains(&transaction) {
                return Err(Error::Execution("mock prepare failed".into()));
            }
            Ok(())
        }

        fn retain_provisional_prefixes(
            &mut self,
            _transaction: ExecutionTransactionId,
            _sources: &[Self::SequenceState],
            _branches: &mut [Self::SequenceState],
            _executed_rows: &[usize],
            _retained_rows: &[usize],
        ) -> Result<()> {
            Ok(())
        }

        fn execute_multi_session_batch_progress(
            &mut self,
            transaction: ExecutionTransactionId,
            _states: &mut [Self::SequenceState],
            _batch: &ExecutionBatch,
        ) -> Result<MultiSessionBatchProgress> {
            self.executes.push(transaction);
            self.next_progress(transaction)
        }

        fn resume_multi_session_batch(
            &mut self,
            transaction: ExecutionTransactionId,
            _states: &mut [Self::SequenceState],
            _batch: &ExecutionBatch,
            continuation: ContinuationId,
            _leases: ExpertLeaseSet,
        ) -> Result<MultiSessionBatchProgress> {
            self.resumes.push((transaction, continuation));
            self.next_progress(transaction)
        }

        fn cancel_multi_session_batch(
            &mut self,
            transaction: ExecutionTransactionId,
            _states: &mut [Self::SequenceState],
            continuation: ContinuationId,
        ) -> BatchContinuationCancelOutcome {
            self.cancels.push((transaction, continuation));
            match self
                .cancel_outcomes
                .get_mut(&transaction)
                .and_then(VecDeque::pop_front)
                .unwrap_or(MockCancelOutcome::Cancelled)
            {
                MockCancelOutcome::Cancelled => BatchContinuationCancelOutcome::Cancelled,
                MockCancelOutcome::StillActive => BatchContinuationCancelOutcome::StillActive(
                    Error::Execution("mock cancellation is still active".into()),
                ),
                MockCancelOutcome::Quiesced => BatchContinuationCancelOutcome::Quiesced(
                    Error::Execution("mock cancellation cleanup failed".into()),
                ),
            }
        }

        fn commit_multi_session_batch(
            &mut self,
            transaction: ExecutionTransactionId,
            _states: &mut [Self::SequenceState],
        ) -> Result<()> {
            self.commits.push(transaction);
            Ok(())
        }

        fn rollback_multi_session_batch(
            &mut self,
            transaction: ExecutionTransactionId,
            _states: &mut [Self::SequenceState],
        ) -> Result<()> {
            self.rollbacks.push(transaction);
            if self.fail_rollbacks.contains(&transaction) {
                Err(Error::Execution("mock backend rollback failed".into()))
            } else {
                Ok(())
            }
        }

        fn multi_session_capabilities(&self) -> ExecutionCapabilities {
            ExecutionCapabilities {
                max_batch_tokens: 16,
                max_sequences: 4,
                max_prefill_query_tokens_per_sequence: 16,
                max_decode_query_tokens_per_sequence: 1,
                max_top_k: NonZeroU32::new(8),
                supports_prefill: true,
                supports_decode: true,
                supports_mixed: true,
                full_logits_width: None,
                kv_binding_mode: ferrule_common::execution::KvBindingMode::None,
                logits_row_policy: ferrule_common::execution::LogitsRowPolicy::LastPerSequence,
            }
        }
    }

    fn transaction(value: u64) -> ExecutionTransactionId {
        ExecutionTransactionId::new(value).unwrap()
    }

    fn continuation(value: u64) -> ContinuationId {
        ContinuationId::new(value)
    }

    fn dependencies(continuation: ContinuationId) -> DependencySet {
        DependencySet::new([LogicalDependency::operation_retired(OperationId::new(
            continuation.get(),
        ))
        .unwrap()])
        .unwrap()
    }

    fn leases(continuation: ContinuationId) -> ExpertLeaseSet {
        ExpertLeaseSet::new(
            [],
            [],
            MappingEpoch::new(continuation.get()),
            DispatchFenceContract::new(
                OperationId::new(continuation.get()),
                FenceId::new(continuation.get()),
                BackendId::new(0),
                DeviceId::new(0),
            ),
        )
        .unwrap()
    }

    fn waiting(
        transaction: ExecutionTransactionId,
        continuation: ContinuationId,
    ) -> MultiSessionBatchProgress {
        MultiSessionBatchProgress::Waiting(
            PendingModelProgress::new(transaction, continuation, dependencies(continuation))
                .unwrap(),
        )
    }

    fn complete(token: u32) -> MultiSessionBatchProgress {
        MultiSessionBatchProgress::Complete(ExecutionOutput::new(vec![LogitsRow::new(
            0,
            LogitsOutput::TopK(vec![TokenLogit::new(token, 1.0)]),
        )]))
    }

    fn decode_batch(token: u32) -> (Vec<MockSequenceState>, ExecutionBatch) {
        let states = vec![MockSequenceState { position: 10 }];
        let batch = ExecutionBatch::new(
            ForwardMode::Decode,
            vec![token],
            vec![10],
            vec![None],
            vec![LogitsRequest::TopK(NonZeroU32::new(5).unwrap())],
            vec![ExecutionSequence::new(
                StateSlot::new(0),
                ForwardPhase::Decode,
                0..1,
                10,
                11,
                0..0,
            )],
            Vec::new(),
        );
        (states, batch)
    }

    #[test]
    fn transactions_wait_together_and_finish_in_reverse_without_a_yield_loop() {
        let transaction_a = transaction(1);
        let transaction_b = transaction(2);
        let continuation_a = continuation(101);
        let continuation_b = continuation(102);
        let mut runner = MockMultiSessionRunner::default();
        runner.push_progress(transaction_a, Ok(waiting(transaction_a, continuation_a)));
        runner.push_progress(transaction_a, Ok(complete(11)));
        runner.push_progress(transaction_b, Ok(waiting(transaction_b, continuation_b)));
        runner.push_progress(transaction_b, Ok(complete(12)));
        let mut executor = NativeMultiSessionExecutor::new(runner);
        let (mut states_a, batch_a) = decode_batch(21);
        let (mut states_b, batch_b) = decode_batch(22);

        assert!(matches!(
            executor
                .begin_resumable_batch_with_kv(transaction_a, &mut states_a, &batch_a, &[])
                .unwrap(),
            NativeBatchExecutionProgress::Waiting(_)
        ));
        assert!(matches!(
            executor
                .begin_resumable_batch_with_kv(transaction_b, &mut states_b, &batch_b, &[])
                .unwrap(),
            NativeBatchExecutionProgress::Waiting(_)
        ));
        assert!(executor.runner().resumes.is_empty());
        assert!(executor.has_transactions());
        assert_eq!(
            executor.transaction_ids().collect::<HashSet<_>>(),
            HashSet::from([transaction_a, transaction_b])
        );

        assert!(matches!(
            executor
                .resume_resumable_batch(
                    transaction_b,
                    &mut states_b,
                    &batch_b,
                    continuation_b,
                    leases(continuation_b),
                )
                .unwrap(),
            NativeBatchExecutionProgress::Complete(_)
        ));
        executor
            .commit_prepared_batch(transaction_b, &mut states_b)
            .unwrap();
        assert_eq!(
            executor
                .pending_model_progress(transaction_a)
                .unwrap()
                .continuation(),
            continuation_a
        );

        executor
            .resume_resumable_batch(
                transaction_a,
                &mut states_a,
                &batch_a,
                continuation_a,
                leases(continuation_a),
            )
            .unwrap();
        executor
            .commit_prepared_batch(transaction_a, &mut states_a)
            .unwrap();
        assert_eq!(
            executor.runner().commits,
            vec![transaction_b, transaction_a]
        );
        assert!(!executor.has_transactions());
    }

    #[test]
    fn wrong_transaction_never_takes_another_transactions_continuation() {
        let transaction_a = transaction(3);
        let transaction_b = transaction(4);
        let continuation_a = continuation(103);
        let continuation_b = continuation(104);
        let mut runner = MockMultiSessionRunner::default();
        runner.push_progress(transaction_a, Ok(waiting(transaction_a, continuation_a)));
        runner.push_progress(transaction_b, Ok(waiting(transaction_b, continuation_b)));
        let mut executor = NativeMultiSessionExecutor::new(runner);
        let (mut states_a, batch_a) = decode_batch(31);
        let (mut states_b, batch_b) = decode_batch(32);
        executor
            .begin_resumable_batch_with_kv(transaction_a, &mut states_a, &batch_a, &[])
            .unwrap();
        executor
            .begin_resumable_batch_with_kv(transaction_b, &mut states_b, &batch_b, &[])
            .unwrap();

        assert!(
            executor
                .resume_resumable_batch(
                    transaction_b,
                    &mut states_b,
                    &batch_b,
                    continuation_a,
                    leases(continuation_a),
                )
                .unwrap_err()
                .to_string()
                .contains("owns continuation")
        );
        assert!(
            executor
                .cancel_resumable_batch(transaction_b, &mut states_b, continuation_a)
                .unwrap_err()
                .to_string()
                .contains("owns continuation")
        );
        assert!(executor.runner().resumes.is_empty());
        assert!(executor.runner().cancels.is_empty());
        assert_eq!(
            executor
                .pending_model_progress(transaction_a)
                .unwrap()
                .continuation(),
            continuation_a
        );
        assert_eq!(
            executor
                .pending_model_progress(transaction_b)
                .unwrap()
                .continuation(),
            continuation_b
        );
    }

    #[test]
    fn invalid_waiting_transaction_is_cancelled_and_rolled_back_atomically() {
        let expected = transaction(17);
        let reported = transaction(18);
        let continuation = continuation(117);
        let mut runner = MockMultiSessionRunner::default();
        runner.push_progress(expected, Ok(waiting(reported, continuation)));
        let mut executor = NativeMultiSessionExecutor::new(runner);
        let (mut states, batch) = decode_batch(33);

        let error = executor
            .begin_resumable_batch_with_kv(expected, &mut states, &batch, &[])
            .unwrap_err();

        assert!(error.to_string().contains("while executing transaction"));
        assert_eq!(executor.runner().cancels, vec![(expected, continuation)]);
        assert_eq!(executor.runner().rollbacks, vec![expected]);
        assert!(executor.pending_model_progress(expected).is_none());
        assert!(!executor.has_transactions());
        assert!(executor.is_poisoned());
    }

    #[test]
    fn invalid_waiting_transaction_still_active_retains_cancellable_ownership() {
        let expected = transaction(19);
        let reported = transaction(20);
        let continuation = continuation(119);
        let mut runner = MockMultiSessionRunner::default();
        runner.push_progress(expected, Ok(waiting(reported, continuation)));
        runner.push_cancel_outcome(expected, MockCancelOutcome::StillActive);
        runner.push_cancel_outcome(expected, MockCancelOutcome::Cancelled);
        let mut executor = NativeMultiSessionExecutor::new(runner);
        let (mut states, batch) = decode_batch(34);

        let error = executor
            .begin_resumable_batch_with_kv(expected, &mut states, &batch, &[])
            .unwrap_err();

        assert!(error.to_string().contains("did not quiesce"));
        let pending = executor.pending_model_progress(expected).unwrap();
        assert_eq!(pending.transaction(), expected);
        assert_eq!(pending.continuation(), continuation);
        assert!(executor.runner().rollbacks.is_empty());

        executor
            .cancel_resumable_batch(expected, &mut states, continuation)
            .unwrap();
        assert_eq!(executor.runner().rollbacks, vec![expected]);
        assert!(!executor.has_transactions());
    }

    #[test]
    fn cancelling_one_transaction_does_not_disturb_another() {
        let transaction_a = transaction(5);
        let transaction_b = transaction(6);
        let continuation_a = continuation(105);
        let continuation_b = continuation(106);
        let mut runner = MockMultiSessionRunner::default();
        runner.push_progress(transaction_a, Ok(waiting(transaction_a, continuation_a)));
        runner.push_progress(transaction_b, Ok(waiting(transaction_b, continuation_b)));
        runner.push_progress(transaction_b, Ok(complete(16)));
        let mut executor = NativeMultiSessionExecutor::new(runner);
        let (mut states_a, batch_a) = decode_batch(41);
        let (mut states_b, batch_b) = decode_batch(42);
        executor
            .begin_resumable_batch_with_kv(transaction_a, &mut states_a, &batch_a, &[])
            .unwrap();
        executor
            .begin_resumable_batch_with_kv(transaction_b, &mut states_b, &batch_b, &[])
            .unwrap();

        executor
            .cancel_resumable_batch(transaction_a, &mut states_a, continuation_a)
            .unwrap();
        assert_eq!(executor.runner().rollbacks, vec![transaction_a]);
        assert!(executor.pending_model_progress(transaction_a).is_none());
        assert_eq!(
            executor
                .pending_model_progress(transaction_b)
                .unwrap()
                .continuation(),
            continuation_b
        );
        assert!(!executor.is_poisoned());

        executor
            .resume_resumable_batch(
                transaction_b,
                &mut states_b,
                &batch_b,
                continuation_b,
                leases(continuation_b),
            )
            .unwrap();
        executor
            .commit_prepared_batch(transaction_b, &mut states_b)
            .unwrap();
        assert_eq!(executor.runner().commits, vec![transaction_b]);
    }

    #[test]
    fn still_active_preserves_registry_and_ownership_until_cancelled() {
        let transaction = transaction(7);
        let continuation = continuation(107);
        let mut runner = MockMultiSessionRunner::default();
        runner.push_progress(transaction, Ok(waiting(transaction, continuation)));
        runner.push_cancel_outcome(transaction, MockCancelOutcome::StillActive);
        runner.push_cancel_outcome(transaction, MockCancelOutcome::Cancelled);
        let mut executor = NativeMultiSessionExecutor::new(runner);
        let (mut states, batch) = decode_batch(51);
        executor
            .begin_resumable_batch_with_kv(transaction, &mut states, &batch, &[])
            .unwrap();

        let error = executor
            .cancel_resumable_batch(transaction, &mut states, continuation)
            .unwrap_err();
        assert!(error.to_string().contains("did not confirm quiescence"));
        assert_eq!(
            executor
                .pending_model_progress(transaction)
                .unwrap()
                .continuation(),
            continuation
        );
        assert!(executor.has_transactions());
        assert!(executor.runner().rollbacks.is_empty());
        assert!(!executor.is_poisoned());

        executor
            .cancel_resumable_batch(transaction, &mut states, continuation)
            .unwrap();
        assert_eq!(executor.runner().rollbacks, vec![transaction]);
        assert!(!executor.has_transactions());
    }

    #[test]
    fn quiesced_cleanup_error_still_rolls_back_and_releases_transaction() {
        let transaction = transaction(8);
        let continuation = continuation(108);
        let mut runner = MockMultiSessionRunner::default();
        runner.push_progress(transaction, Ok(waiting(transaction, continuation)));
        runner.push_cancel_outcome(transaction, MockCancelOutcome::Quiesced);
        let mut executor = NativeMultiSessionExecutor::new(runner);
        let (mut states, batch) = decode_batch(61);
        executor
            .begin_resumable_batch_with_kv(transaction, &mut states, &batch, &[])
            .unwrap();

        let error = executor
            .cancel_resumable_batch(transaction, &mut states, continuation)
            .unwrap_err();
        assert!(error.to_string().contains("model work quiesced"));
        assert_eq!(executor.runner().rollbacks, vec![transaction]);
        assert!(!executor.has_transactions());
        assert!(!executor.is_poisoned());
    }

    #[test]
    fn commit_and_rollback_are_invoked_exactly_once() {
        let committed = transaction(9);
        let rolled_back = transaction(10);
        let mut runner = MockMultiSessionRunner::default();
        runner.push_progress(committed, Ok(complete(19)));
        runner.push_progress(rolled_back, Ok(complete(20)));
        let mut executor = NativeMultiSessionExecutor::new(runner);
        let (mut committed_states, committed_batch) = decode_batch(71);
        let (mut rolled_back_states, rolled_back_batch) = decode_batch(72);
        executor
            .begin_resumable_batch_with_kv(committed, &mut committed_states, &committed_batch, &[])
            .unwrap();
        executor
            .begin_resumable_batch_with_kv(
                rolled_back,
                &mut rolled_back_states,
                &rolled_back_batch,
                &[],
            )
            .unwrap();

        executor
            .commit_prepared_batch(committed, &mut committed_states)
            .unwrap();
        assert!(
            executor
                .commit_prepared_batch(committed, &mut committed_states)
                .is_err()
        );
        executor
            .rollback_prepared_batch(rolled_back, &mut rolled_back_states)
            .unwrap();
        assert!(
            executor
                .rollback_prepared_batch(rolled_back, &mut rolled_back_states)
                .is_err()
        );
        assert_eq!(executor.runner().commits, vec![committed]);
        assert_eq!(executor.runner().rollbacks, vec![rolled_back]);
    }

    #[test]
    fn prepare_and_execute_errors_leave_unrelated_transaction_usable() {
        let waiting_transaction = transaction(11);
        let prepare_failure = transaction(12);
        let execute_failure = transaction(13);
        let waiting_continuation = continuation(111);
        let mut runner = MockMultiSessionRunner::default();
        runner.push_progress(
            waiting_transaction,
            Ok(waiting(waiting_transaction, waiting_continuation)),
        );
        runner.fail_prepares.insert(prepare_failure);
        runner.push_progress(
            execute_failure,
            Err(Error::Execution("mock execution failed".into())),
        );
        let mut executor = NativeMultiSessionExecutor::new(runner);
        let (mut waiting_states, waiting_batch) = decode_batch(81);
        let (mut failed_states, failed_batch) = decode_batch(82);
        executor
            .begin_resumable_batch_with_kv(
                waiting_transaction,
                &mut waiting_states,
                &waiting_batch,
                &[],
            )
            .unwrap();

        assert!(
            executor
                .begin_resumable_batch_with_kv(
                    prepare_failure,
                    &mut failed_states,
                    &failed_batch,
                    &[],
                )
                .is_err()
        );
        assert!(executor.runner().rollbacks.is_empty());
        assert!(
            executor
                .begin_resumable_batch_with_kv(
                    execute_failure,
                    &mut failed_states,
                    &failed_batch,
                    &[],
                )
                .is_err()
        );
        assert_eq!(executor.runner().rollbacks, vec![execute_failure]);
        assert!(!executor.is_poisoned());
        assert_eq!(
            executor
                .pending_model_progress(waiting_transaction)
                .unwrap()
                .continuation(),
            waiting_continuation
        );
    }

    #[test]
    fn sequence_ownership_operations_ignore_unrelated_transactions_but_global_operations_do_not() {
        let transaction = transaction(14);
        let continuation = continuation(114);
        let mut runner = MockMultiSessionRunner::default();
        runner.push_progress(transaction, Ok(waiting(transaction, continuation)));
        let mut executor = NativeMultiSessionExecutor::new(runner);
        let (mut states, batch) = decode_batch(91);
        executor
            .begin_resumable_batch_with_kv(transaction, &mut states, &batch, &[])
            .unwrap();

        let state = executor.create_sequence_state().unwrap();
        let fork = executor
            .fork_sequence_state_from(&state, state.position)
            .unwrap();
        executor.release_sequence_state(fork).unwrap();
        assert_eq!(executor.runner().sequences_created, 1);
        assert_eq!(executor.runner().sequences_forked, 1);
        assert_eq!(executor.runner().sequences_released, 1);
        assert!(executor.reset().is_err());
        assert!(executor.configure_kv_page_capacity(16).is_err());

        let failure = executor.try_into_runner().unwrap_err();
        let (error, mut executor) = *failure;
        assert!(error.to_string().contains("active execution transactions"));
        executor
            .cancel_resumable_batch(transaction, &mut states, continuation)
            .unwrap();
        executor.reset().unwrap();
        executor.configure_kv_page_capacity(16).unwrap();
        assert!(executor.try_into_runner().is_ok());
    }

    #[test]
    fn rollback_failure_poisons_and_retains_only_the_failed_transaction() {
        let failed = transaction(15);
        let unaffected = transaction(16);
        let mut runner = MockMultiSessionRunner::default();
        runner.fail_rollbacks.insert(failed);
        runner.push_progress(
            failed,
            Err(Error::Execution("mock execution failed".into())),
        );
        runner.push_progress(unaffected, Ok(complete(26)));
        let mut executor = NativeMultiSessionExecutor::new(runner);
        let (mut failed_states, failed_batch) = decode_batch(101);
        let (mut unaffected_states, unaffected_batch) = decode_batch(102);
        executor
            .begin_resumable_batch_with_kv(
                unaffected,
                &mut unaffected_states,
                &unaffected_batch,
                &[],
            )
            .unwrap();

        let error = executor
            .begin_resumable_batch_with_kv(failed, &mut failed_states, &failed_batch, &[])
            .unwrap_err();
        assert!(error.to_string().contains("backend rollback also failed"));
        assert!(executor.is_poisoned());
        assert_eq!(
            executor.transaction_ids().collect::<HashSet<_>>(),
            HashSet::from([failed, unaffected])
        );
        assert!(
            executor
                .commit_prepared_batch(unaffected, &mut unaffected_states)
                .is_ok()
        );
        assert_eq!(executor.transaction_ids().collect::<Vec<_>>(), vec![failed]);
    }
}
