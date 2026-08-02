//! Stateless native multi-session backend adapter.
//!
//! The driver owns logical transaction payloads; the model backend owns submitted
//! physical work. This adapter validates the neutral batch contract and forwards
//! calls without maintaining a third lifecycle registry.

use ferrule_common::execution::{
    ExecutionBatch, ExecutionCapabilities, ExecutionOutput, ExecutionTransactionId,
    KvReservationView,
};
use ferrule_common::{ContinuationId, ResidencyLeaseSet};

use crate::{Error, Result};
use ferrule_model::{
    MultiSessionBatchProgress, MultiSessionRunner, PendingModelProgress, TransactionEndIntent,
    TransactionEndProgress,
};

use crate::expert_residency::ExpertResidencyController;

#[derive(Debug, Clone, PartialEq)]
pub enum NativeBatchExecutionProgress {
    Complete(ExecutionOutput),
    Waiting(PendingModelProgress),
}

/// Thin owner-thread adapter over a `MultiSessionRunner`.
///
/// A successful `prepare_batch_with_kv` call transfers backend transaction
/// ownership to the caller. Every later error leaves that ownership with the
/// caller, which must drive `end_transaction`.
pub struct NativeMultiSessionExecutor<R: MultiSessionRunner> {
    runner: R,
    capabilities: ExecutionCapabilities,
    expert_residency_initialized: bool,
}

impl<R: MultiSessionRunner> NativeMultiSessionExecutor<R> {
    pub fn new(runner: R) -> Self {
        let capabilities = runner.multi_session_capabilities();
        Self {
            runner,
            capabilities,
            expert_residency_initialized: false,
        }
    }

    pub fn capabilities(&self) -> &ExecutionCapabilities {
        &self.capabilities
    }

    pub(crate) fn runner(&self) -> &R {
        &self.runner
    }

    pub(crate) fn runner_mut(&mut self) -> &mut R {
        &mut self.runner
    }

    pub fn configure_kv_page_capacity(&mut self, max_pages: usize) -> Result<()> {
        self.runner.configure_kv_page_capacity(max_pages)?;
        self.capabilities = self.runner.multi_session_capabilities();
        Ok(())
    }

    pub fn into_runner(self) -> R {
        self.runner
    }

    pub fn reset(&mut self) -> Result<()> {
        Ok(self.runner.reset_session()?)
    }

    pub fn create_sequence_state(&mut self) -> Result<R::SequenceState> {
        Ok(self.runner.create_sequence_state()?)
    }

    pub fn fork_sequence_state_from(
        &mut self,
        source: &R::SequenceState,
        expected_position: usize,
    ) -> Result<R::SequenceState> {
        Ok(self
            .runner
            .fork_sequence_state_from(source, expected_position)?)
    }

    pub fn reset_sequence_state(&mut self, state: &mut R::SequenceState) -> Result<()> {
        Ok(self.runner.reset_sequence_state(state)?)
    }

    pub fn release_sequence_state(&mut self, state: R::SequenceState) -> Result<()> {
        Ok(self.runner.release_sequence_state(state)?)
    }

    pub fn with_sequence_state<T>(
        &mut self,
        state: &mut R::SequenceState,
        execute: impl FnOnce(&mut R) -> ferrule_common::Result<T>,
    ) -> Result<T> {
        self.ensure_execution_ready()?;
        Ok(self.runner.with_sequence_state(state, execute)?)
    }

    pub fn release_kv_pages(
        &mut self,
        pages: &[ferrule_common::execution::KvPageId],
    ) -> Result<()> {
        Ok(self.runner.release_kv_pages(pages)?)
    }

    pub fn preempt_kv_pages(
        &mut self,
        pages: &[ferrule_common::execution::KvPageId],
    ) -> Result<()> {
        Ok(self.runner.preempt_kv_pages(pages)?)
    }

    pub fn restore_kv_pages(
        &mut self,
        pages: &[ferrule_common::execution::KvPageId],
    ) -> Result<()> {
        Ok(self.runner.restore_kv_pages(pages)?)
    }

    pub fn end_transaction(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [R::SequenceState],
        intent: TransactionEndIntent,
    ) -> Result<TransactionEndProgress> {
        Ok(self.runner.end_transaction(transaction, states, intent)?)
    }

    pub fn retain_prepared_prefixes(
        &mut self,
        transaction: ExecutionTransactionId,
        sources: &[R::SequenceState],
        branches: &mut [R::SequenceState],
        executed_rows: &[usize],
        retained_rows: &[usize],
    ) -> Result<()> {
        let sequence_count = sources.len();
        if sequence_count == 0
            || branches.len() != sequence_count
            || executed_rows.len() != sequence_count
            || retained_rows.len() != sequence_count
        {
            return Err(Error::InvalidRequest {
                message: format!(
                    "provisional prefix cohort shape mismatch: sources={} branches={} executed={} retained={}",
                    sequence_count,
                    branches.len(),
                    executed_rows.len(),
                    retained_rows.len()
                ),
            });
        }
        for (sequence, (&executed, &retained)) in
            executed_rows.iter().zip(retained_rows).enumerate()
        {
            if executed == 0 || retained == 0 || retained > executed {
                return Err(Error::InvalidRequest {
                    message: format!(
                        "invalid provisional prefix for sequence {sequence}: retained={retained} executed={executed}"
                    ),
                });
            }
        }
        Ok(self.runner.retain_provisional_prefixes(
            transaction,
            sources,
            branches,
            executed_rows,
            retained_rows,
        )?)
    }

    /// Prepare backend resources. On success the caller owns a live transaction,
    /// even if the following execute call fails.
    pub fn prepare_batch_with_kv(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [R::SequenceState],
        batch: &ExecutionBatch,
        kv_reservations: &[KvReservationView],
    ) -> Result<()> {
        self.ensure_execution_ready()?;
        batch.validate(states.len(), &self.capabilities)?;
        Ok(self
            .runner
            .prepare_multi_session_batch(transaction, states, batch, kv_reservations)?)
    }

    pub fn execute_prepared_batch(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [R::SequenceState],
        batch: &ExecutionBatch,
    ) -> Result<NativeBatchExecutionProgress> {
        batch.validate(states.len(), &self.capabilities)?;
        let progress =
            self.runner
                .execute_multi_session_batch_progress(transaction, states, batch)?;
        Self::validate_progress(transaction, batch, &self.capabilities, progress)
    }

    pub fn resume_prepared_batch(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [R::SequenceState],
        batch: &ExecutionBatch,
        continuation: ContinuationId,
        leases: ResidencyLeaseSet,
    ) -> Result<NativeBatchExecutionProgress> {
        batch.validate(states.len(), &self.capabilities)?;
        let progress = self.runner.resume_multi_session_batch(
            transaction,
            states,
            batch,
            continuation,
            leases,
        )?;
        Self::validate_progress(transaction, batch, &self.capabilities, progress)
    }

    fn validate_progress(
        transaction: ExecutionTransactionId,
        batch: &ExecutionBatch,
        capabilities: &ExecutionCapabilities,
        progress: MultiSessionBatchProgress,
    ) -> Result<NativeBatchExecutionProgress> {
        match progress {
            MultiSessionBatchProgress::Complete(output) => {
                output.validate_with_capabilities(batch, capabilities)?;
                Ok(NativeBatchExecutionProgress::Complete(output))
            }
            MultiSessionBatchProgress::Waiting(pending) => {
                if pending.transaction() != transaction {
                    return Err(Error::InvalidRequest {
                        message: format!(
                            "runner returned waiting progress for transaction {:?} while executing transaction {transaction:?}",
                            pending.transaction()
                        ),
                    });
                }
                Ok(NativeBatchExecutionProgress::Waiting(pending))
            }
        }
    }

    fn ensure_execution_ready(&mut self) -> Result<()> {
        if self.expert_residency_initialized || self.runner.expert_residency_control_installed() {
            self.expert_residency_initialized = true;
            return Ok(());
        }
        let Some(requirements) = self.runner.expert_residency_requirements() else {
            self.expert_residency_initialized = true;
            return Ok(());
        };
        let control = ExpertResidencyController::with_requirements(requirements)?;
        self.runner
            .install_expert_residency_control(Box::new(control))?;
        self.expert_residency_initialized = true;
        Ok(())
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
    use ferrule_common::{DependencySet, LogicalDependency, OperationId};
    use ferrule_common::{Error as ModelError, Result as ModelResult};
    use ferrule_model::{ModelInfo, ModelRunner};

    #[derive(Debug, Default)]
    struct MockSequenceState {
        position: usize,
    }

    #[derive(Debug)]
    enum MockCancelOutcome {
        Complete,
        Pending,
        Failed,
    }

    #[derive(Debug, Default)]
    struct MockMultiSessionRunner {
        progress: HashMap<ExecutionTransactionId, VecDeque<ModelResult<MultiSessionBatchProgress>>>,
        cancel_outcomes: HashMap<ExecutionTransactionId, VecDeque<MockCancelOutcome>>,
        fail_prepares: HashSet<ExecutionTransactionId>,
        fail_rollbacks: HashSet<ExecutionTransactionId>,
        prepares: Vec<ExecutionTransactionId>,
        executes: Vec<ExecutionTransactionId>,
        resumes: Vec<(ExecutionTransactionId, ContinuationId)>,

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
            progress: ModelResult<MultiSessionBatchProgress>,
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
        ) -> ModelResult<MultiSessionBatchProgress> {
            self.progress
                .get_mut(&transaction)
                .and_then(VecDeque::pop_front)
                .unwrap_or_else(|| {
                    Err(ModelError::Execution {
                        message: format!("mock has no progress for transaction {transaction:?}"),
                    })
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

        fn encode(&self, text: &str) -> ModelResult<Vec<u32>> {
            Ok(text.bytes().map(u32::from).collect())
        }

        fn decode(&self, tokens: &[u32]) -> ModelResult<String> {
            Ok(tokens
                .iter()
                .map(|token| char::from_u32(*token).unwrap_or('?'))
                .collect())
        }

        fn reset_session(&mut self) -> ModelResult<()> {
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
            execute: impl FnOnce(&mut Self) -> ModelResult<T>,
        ) -> ModelResult<T> {
            execute(self)
        }

        fn create_sequence_state(&mut self) -> ModelResult<Self::SequenceState> {
            self.sequences_created += 1;
            Ok(MockSequenceState::default())
        }

        fn fork_sequence_state(&mut self) -> ModelResult<Self::SequenceState> {
            self.sequences_forked += 1;
            Ok(MockSequenceState::default())
        }

        fn fork_sequence_state_from(
            &mut self,
            source: &Self::SequenceState,
            expected_position: usize,
        ) -> ModelResult<Self::SequenceState> {
            if source.position != expected_position {
                return Err(ModelError::Execution {
                    message: "mock exact fork position mismatch".into(),
                });
            }
            self.sequences_forked += 1;
            Ok(MockSequenceState {
                position: source.position,
            })
        }

        fn reset_sequence_state(&mut self, state: &mut Self::SequenceState) -> ModelResult<()> {
            state.position = 0;
            Ok(())
        }

        fn release_sequence_state(&mut self, _state: Self::SequenceState) -> ModelResult<()> {
            self.sequences_released += 1;
            Ok(())
        }

        fn configure_kv_page_capacity(&mut self, _max_pages: usize) -> ModelResult<()> {
            Ok(())
        }

        fn release_kv_pages(
            &mut self,
            _pages: &[ferrule_common::execution::KvPageId],
        ) -> ModelResult<()> {
            Ok(())
        }

        fn preempt_kv_pages(
            &mut self,
            _pages: &[ferrule_common::execution::KvPageId],
        ) -> ModelResult<()> {
            Ok(())
        }

        fn restore_kv_pages(
            &mut self,
            _pages: &[ferrule_common::execution::KvPageId],
        ) -> ModelResult<()> {
            Ok(())
        }

        fn prepare_multi_session_batch(
            &mut self,
            transaction: ExecutionTransactionId,
            _states: &mut [Self::SequenceState],
            _batch: &ExecutionBatch,
            _kv_reservations: &[KvReservationView],
        ) -> ModelResult<()> {
            self.prepares.push(transaction);
            if self.fail_prepares.contains(&transaction) {
                return Err(ModelError::Execution {
                    message: "mock prepare failed".into(),
                });
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
        ) -> ModelResult<()> {
            Ok(())
        }

        fn execute_multi_session_batch_progress(
            &mut self,
            transaction: ExecutionTransactionId,
            _states: &mut [Self::SequenceState],
            _batch: &ExecutionBatch,
        ) -> ModelResult<MultiSessionBatchProgress> {
            self.executes.push(transaction);
            self.next_progress(transaction)
        }

        fn resume_multi_session_batch(
            &mut self,
            transaction: ExecutionTransactionId,
            _states: &mut [Self::SequenceState],
            _batch: &ExecutionBatch,
            continuation: ContinuationId,
            _leases: ResidencyLeaseSet,
        ) -> ModelResult<MultiSessionBatchProgress> {
            self.resumes.push((transaction, continuation));
            self.next_progress(transaction)
        }

        fn end_transaction(
            &mut self,
            transaction: ExecutionTransactionId,
            _states: &mut [Self::SequenceState],
            intent: TransactionEndIntent,
        ) -> ModelResult<TransactionEndProgress> {
            match intent {
                TransactionEndIntent::Publish => {
                    self.commits.push(transaction);
                    Ok(TransactionEndProgress::Complete)
                }
                TransactionEndIntent::Abort => {
                    self.rollbacks.push(transaction);
                    if self.fail_rollbacks.contains(&transaction) {
                        return Err(ModelError::Execution {
                            message: "mock backend rollback failed".into(),
                        });
                    }
                    match self
                        .cancel_outcomes
                        .get_mut(&transaction)
                        .and_then(VecDeque::pop_front)
                        .unwrap_or(MockCancelOutcome::Complete)
                    {
                        MockCancelOutcome::Complete => Ok(TransactionEndProgress::Complete),
                        MockCancelOutcome::Pending => Ok(TransactionEndProgress::Pending),
                        MockCancelOutcome::Failed => Err(ModelError::Execution {
                            message: "mock cancellation cleanup failed".into(),
                        }),
                    }
                }
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
    fn prepare_and_execute_are_explicit_owner_steps() {
        let transaction = transaction(1);
        let mut runner = MockMultiSessionRunner::default();
        runner.push_progress(transaction, Ok(complete(11)));
        let mut executor = NativeMultiSessionExecutor::new(runner);
        let (mut states, batch) = decode_batch(31);

        executor
            .prepare_batch_with_kv(transaction, &mut states, &batch, &[])
            .unwrap();
        assert_eq!(executor.runner().prepares, vec![transaction]);
        assert!(matches!(
            executor
                .execute_prepared_batch(transaction, &mut states, &batch)
                .unwrap(),
            NativeBatchExecutionProgress::Complete(_)
        ));
        assert!(executor.runner().commits.is_empty());
        assert!(executor.runner().rollbacks.is_empty());
    }

    #[test]
    fn execute_error_leaves_terminalization_to_the_owner() {
        let transaction = transaction(2);
        let mut runner = MockMultiSessionRunner::default();
        runner.push_progress(
            transaction,
            Err(ModelError::Execution {
                message: "mock execution failed".into(),
            }),
        );
        let mut executor = NativeMultiSessionExecutor::new(runner);
        let (mut states, batch) = decode_batch(32);

        executor
            .prepare_batch_with_kv(transaction, &mut states, &batch, &[])
            .unwrap();
        assert!(
            executor
                .execute_prepared_batch(transaction, &mut states, &batch)
                .is_err()
        );
        assert!(executor.runner().rollbacks.is_empty());
        assert_eq!(
            executor
                .end_transaction(transaction, &mut states, TransactionEndIntent::Abort)
                .unwrap(),
            TransactionEndProgress::Complete
        );
        assert_eq!(executor.runner().rollbacks, vec![transaction]);
    }

    #[test]
    fn abort_pending_is_normal_progress_and_retryable() {
        let transaction = transaction(3);
        let mut runner = MockMultiSessionRunner::default();
        runner.push_cancel_outcome(transaction, MockCancelOutcome::Pending);
        runner.push_cancel_outcome(transaction, MockCancelOutcome::Complete);
        let mut executor = NativeMultiSessionExecutor::new(runner);
        let (mut states, _) = decode_batch(33);

        assert_eq!(
            executor
                .end_transaction(transaction, &mut states, TransactionEndIntent::Abort)
                .unwrap(),
            TransactionEndProgress::Pending
        );
        assert_eq!(
            executor
                .end_transaction(transaction, &mut states, TransactionEndIntent::Abort)
                .unwrap(),
            TransactionEndProgress::Complete
        );
        assert_eq!(executor.runner().rollbacks, vec![transaction, transaction]);
    }

    #[test]
    fn fatal_terminal_error_is_not_reclassified_as_pending() {
        let transaction = transaction(4);
        let mut runner = MockMultiSessionRunner::default();
        runner.push_cancel_outcome(transaction, MockCancelOutcome::Failed);
        let mut executor = NativeMultiSessionExecutor::new(runner);
        let (mut states, _) = decode_batch(34);

        let error = executor
            .end_transaction(transaction, &mut states, TransactionEndIntent::Abort)
            .unwrap_err();
        assert!(error.to_string().contains("cleanup failed"));
    }

    #[test]
    fn waiting_progress_is_transaction_qualified() {
        let expected = transaction(5);
        let reported = transaction(6);
        let continuation = continuation(105);
        let mut runner = MockMultiSessionRunner::default();
        runner.push_progress(expected, Ok(waiting(reported, continuation)));
        let mut executor = NativeMultiSessionExecutor::new(runner);
        let (mut states, batch) = decode_batch(35);
        executor
            .prepare_batch_with_kv(expected, &mut states, &batch, &[])
            .unwrap();

        let error = executor
            .execute_prepared_batch(expected, &mut states, &batch)
            .unwrap_err();
        assert!(error.to_string().contains("while executing transaction"));
    }

    #[test]
    fn transactions_terminalize_independently() {
        let publish = transaction(7);
        let abort = transaction(8);
        let mut executor = NativeMultiSessionExecutor::new(MockMultiSessionRunner::default());
        let (mut publish_states, _) = decode_batch(36);
        let (mut abort_states, _) = decode_batch(37);

        assert_eq!(
            executor
                .end_transaction(publish, &mut publish_states, TransactionEndIntent::Publish,)
                .unwrap(),
            TransactionEndProgress::Complete
        );
        assert_eq!(
            executor
                .end_transaction(abort, &mut abort_states, TransactionEndIntent::Abort)
                .unwrap(),
            TransactionEndProgress::Complete
        );
        assert_eq!(executor.runner().commits, vec![publish]);
        assert_eq!(executor.runner().rollbacks, vec![abort]);
    }
}
