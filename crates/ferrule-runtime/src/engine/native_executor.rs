//! Generic native multi-session executor.
//!
//! This executor wraps any [`MultiSessionRunner`] and implements the neutral
//! `ExecutionBatch` -> `ExecutionOutput` contract for batches containing
//! multiple sequences, ragged prefill chunks, and mixed prefill/decode rows.
//!
//! Each sequence is routed through the runner's `with_sequence_state` swap path
//! so that all sequences share one set of prepared resources (weights, expert
//! residency, arenas).

use std::num::NonZeroU32;

use ferrule_common::execution::{
    ExecutionBatch, ExecutionCapabilities, ExecutionOutput, ForwardMode, ForwardPhase,
    KvReservation, LogitsOutput, LogitsRequest, LogitsRow, StateSlot,
    TokenLogit as ExecutionTokenLogit,
};
use ferrule_common::{Error, Result};
use ferrule_model::{MultiSessionRunner, PrefillMode, TokenLogit};

use crate::expert_residency::ExpertResidencyController;

/// Native multi-session executor wrapping any [`MultiSessionRunner`].
///
/// The executor owns the runner and its default sequence. Additional sequences
/// are managed externally and passed to `execute_batch` through the state slice.
/// Each sequence in the batch is executed through `with_sequence_state`, which
/// swaps the sequence state into the runner for the duration of the call.
///
/// Poison semantics: a model execution or output-contract failure poisons the
/// executor. A poisoned executor rejects further execution until
/// [`reset`](Self::reset) clears the poison.
pub struct NativeMultiSessionExecutor<R: MultiSessionRunner> {
    runner: R,
    capabilities: ExecutionCapabilities,
    poison: Option<PoisonState>,
    batch_prepared: bool,
    expert_residency_initialized: bool,
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
            batch_prepared: false,
            expert_residency_initialized: false,
        }
    }

    /// Returns the truthful capabilities of the native multi-session path.
    pub fn capabilities(&self) -> &ExecutionCapabilities {
        &self.capabilities
    }

    /// Returns a reference to the underlying runner.
    pub fn runner(&self) -> &R {
        &self.runner
    }

    /// Returns a mutable reference to the underlying runner.
    pub fn runner_mut(&mut self) -> &mut R {
        &mut self.runner
    }

    /// Configure backend physical KV capacity and refresh truthful capabilities.
    pub fn configure_kv_page_capacity(&mut self, max_pages: usize) -> Result<()> {
        if self.batch_prepared {
            return Err(Error::Execution(
                "cannot reconfigure KV capacity with an uncommitted backend batch".into(),
            ));
        }
        self.runner.configure_kv_page_capacity(max_pages)?;
        self.capabilities = self.runner.multi_session_capabilities();
        Ok(())
    }

    /// Extracts the runner if the executor is not poisoned.
    pub fn into_runner(self) -> Result<R> {
        if self.batch_prepared {
            return Err(Error::Execution(
                "cannot extract runner with an uncommitted backend batch".into(),
            ));
        }
        if let Some(poison) = self.poison {
            return Err(Error::Execution(format!(
                "cannot extract runner after {} failed: {}",
                poison.operation, poison.cause
            )));
        }
        Ok(self.runner)
    }

    /// Whether a model mutation failed and left state potentially inconsistent.
    pub fn is_poisoned(&self) -> bool {
        self.poison.is_some()
    }

    pub fn poison_operation(&self) -> Option<&'static str> {
        self.poison.as_ref().map(|p| p.operation)
    }

    pub fn poison_cause(&self) -> Option<&str> {
        self.poison.as_ref().map(|p| p.cause.as_str())
    }

    /// Reset the default runner session and clear poison.
    pub fn reset(&mut self) -> Result<()> {
        if self.batch_prepared {
            self.runner.rollback_multi_session_batch()?;
            self.batch_prepared = false;
        }
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

    /// Mark the executor as poisoned after a runtime-side failure (e.g. token
    /// callback error) that may have left sequence state inconsistent.
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
        if self.batch_prepared {
            return Err(Error::Execution(
                "cannot fork a sequence with an uncommitted backend batch".into(),
            ));
        }
        self.runner
            .fork_sequence_state_from(source, expected_position)
    }

    /// Reset a sequence state for reuse with a new logical sequence.
    pub fn reset_sequence_state(&mut self, state: &mut R::SequenceState) -> Result<()> {
        self.runner.reset_sequence_state(state)
    }

    /// Release a sequence state and its physical capacity.
    pub fn release_sequence_state(&mut self, state: R::SequenceState) -> Result<()> {
        self.runner.release_sequence_state(state)
    }

    /// Feed a token into one explicit sequence state.
    pub fn feed_sequence_token(
        &mut self,
        state: &mut R::SequenceState,
        token_id: u32,
    ) -> Result<()> {
        self.ensure_execution_ready()?;
        self.runner
            .with_sequence_state(state, |runner| runner.feed_token(token_id))
    }

    /// Release physical pages whose runtime refcount reached zero.
    pub fn release_kv_pages(
        &mut self,
        pages: &[ferrule_common::execution::KvPageId],
    ) -> Result<()> {
        self.runner.release_kv_pages(pages)
    }

    pub fn preempt_kv_pages(
        &mut self,
        pages: &[ferrule_common::execution::KvPageId],
    ) -> Result<()> {
        self.ensure_ready()?;
        if self.batch_prepared {
            return Err(Error::Execution(
                "cannot preempt KV pages with an uncommitted backend batch".into(),
            ));
        }
        self.runner.preempt_kv_pages(pages)
    }

    pub fn restore_kv_pages(
        &mut self,
        pages: &[ferrule_common::execution::KvPageId],
    ) -> Result<()> {
        self.ensure_ready()?;
        if self.batch_prepared {
            return Err(Error::Execution(
                "cannot restore KV pages with an uncommitted backend batch".into(),
            ));
        }
        self.runner.restore_kv_pages(pages)
    }

    /// Commit the backend resources prepared by the last successful execution.
    pub fn commit_prepared_batch(&mut self) -> Result<()> {
        if !self.batch_prepared {
            return Ok(());
        }
        match self.runner.commit_multi_session_batch() {
            Ok(()) => {
                self.batch_prepared = false;
                Ok(())
            }
            Err(error) => {
                self.record_poison("backend_kv_commit", &error);
                Err(error)
            }
        }
    }

    /// Roll back the backend resources prepared by the current execution.
    pub fn rollback_prepared_batch(&mut self) -> Result<()> {
        if !self.batch_prepared {
            return Ok(());
        }
        match self.runner.rollback_multi_session_batch() {
            Ok(()) => {
                self.batch_prepared = false;
                Ok(())
            }
            Err(error) => {
                self.record_poison("backend_kv_rollback", &error);
                Err(error)
            }
        }
    }

    /// Prepare paged KV for a diagnostic operation against explicit sequence states.
    ///
    /// Unlike [`execute_batch_with_kv`](Self::execute_batch_with_kv), the caller
    /// supplies the model operation so model-family diagnostics can capture
    /// intermediate state while retaining the runtime's normal KV transaction.
    /// A successful call leaves the backend transaction prepared; the caller must
    /// commit or roll it back together with the authoritative page manager.
    pub fn execute_diagnostic_batch_with_kv<T>(
        &mut self,
        states: &mut [R::SequenceState],
        batch: &ExecutionBatch,
        kv_reservations: &[KvReservation],
        execute: impl FnOnce(&mut R, &mut [R::SequenceState]) -> Result<T>,
    ) -> Result<T> {
        self.ensure_execution_ready()?;
        if self.batch_prepared {
            return Err(Error::Execution(
                "native executor has an uncommitted backend batch".into(),
            ));
        }
        batch.validate(states.len(), &self.capabilities)?;
        self.batch_prepared =
            self.runner
                .prepare_multi_session_batch(states, batch, kv_reservations)?;

        match execute(&mut self.runner, states) {
            Ok(output) => Ok(output),
            Err(error) => {
                let rollback = self.rollback_prepared_batch();
                self.record_poison("diagnostic_model_execution", &error);
                match rollback {
                    Ok(()) => Err(error),
                    Err(rollback_error) => Err(Error::Execution(format!(
                        "diagnostic model execution failed ({error}); backend rollback also failed ({rollback_error})"
                    ))),
                }
            }
        }
    }

    /// Execute a batch with multiple sequences.
    ///
    /// Each sequence in the batch is executed through the runner's
    /// `with_sequence_state` swap path. Output rows are correlated by
    /// `input_row` and collected into one [`ExecutionOutput`].
    ///
    /// The batch is validated against the executor's capabilities before any
    /// mutation. A model execution or output-contract failure poisons the
    /// executor.
    pub fn execute_batch(
        &mut self,
        states: &mut [R::SequenceState],
        batch: &ExecutionBatch,
    ) -> Result<ExecutionOutput> {
        self.execute_batch_with_kv(states, batch, &[])
    }

    /// Execute using the runtime's authoritative provisional KV reservations.
    pub fn execute_batch_with_kv(
        &mut self,
        states: &mut [R::SequenceState],
        batch: &ExecutionBatch,
        kv_reservations: &[KvReservation],
    ) -> Result<ExecutionOutput> {
        self.ensure_execution_ready()?;
        if self.batch_prepared {
            return Err(Error::Execution(
                "native executor has an uncommitted backend batch".into(),
            ));
        }
        batch.validate(states.len(), &self.capabilities)?;
        self.batch_prepared =
            self.runner
                .prepare_multi_session_batch(states, batch, kv_reservations)?;

        let result = match self.runner.execute_multi_session_batch(states, batch) {
            Ok(Some(output)) => Ok(output),
            Ok(None) => match batch.mode() {
                ForwardMode::Prefill => self.execute_prefill_batch(states, batch),
                ForwardMode::Decode => self.execute_decode_batch(states, batch),
                ForwardMode::Mixed => self.execute_mixed_batch(states, batch),
            },
            Err(error) => Err(error),
        };

        match result {
            Ok(output) => {
                if let Err(error) = output.validate_with_capabilities(batch, &self.capabilities) {
                    let rollback = self.rollback_prepared_batch();
                    self.record_poison("output_contract", &error);
                    return match rollback {
                        Ok(()) => Err(error),
                        Err(rollback_error) => Err(Error::Execution(format!(
                            "output contract failed ({error}); backend rollback also failed ({rollback_error})"
                        ))),
                    };
                }
                Ok(output)
            }
            Err(error) => {
                let rollback = self.rollback_prepared_batch();
                self.record_poison("model_execution", &error);
                match rollback {
                    Ok(()) => Err(error),
                    Err(rollback_error) => Err(Error::Execution(format!(
                        "model execution failed ({error}); backend rollback also failed ({rollback_error})"
                    ))),
                }
            }
        }
    }

    fn execute_prefill_batch(
        &mut self,
        states: &mut [R::SequenceState],
        batch: &ExecutionBatch,
    ) -> Result<ExecutionOutput> {
        let mut output_rows = Vec::new();

        for sequence in batch.sequences() {
            let state_index = state_slot_index(sequence.state_slot, states.len())?;
            let state = &mut states[state_index];

            let (query_start, query_end) = query_range(sequence)?;
            let token_ids = &batch.token_ids()[query_start..query_end];
            let last_row = query_end - 1;
            let logits_request = batch.logits()[last_row];

            execute_prefill_sequence(
                &mut self.runner,
                state,
                token_ids,
                logits_request,
                last_row,
                &mut output_rows,
            )?;
        }

        Ok(ExecutionOutput::new(output_rows))
    }

    fn execute_decode_batch(
        &mut self,
        states: &mut [R::SequenceState],
        batch: &ExecutionBatch,
    ) -> Result<ExecutionOutput> {
        let mut output_rows = Vec::new();

        for sequence in batch.sequences() {
            let state_index = state_slot_index(sequence.state_slot, states.len())?;
            let state = &mut states[state_index];

            let (query_start, _query_end) = query_range(sequence)?;
            let row = query_start;
            let token_id = batch.token_ids()[row];
            let logits_request = batch.logits()[row];

            execute_decode_sequence(
                &mut self.runner,
                state,
                token_id,
                logits_request,
                row,
                &mut output_rows,
            )?;
        }

        Ok(ExecutionOutput::new(output_rows))
    }

    fn execute_mixed_batch(
        &mut self,
        states: &mut [R::SequenceState],
        batch: &ExecutionBatch,
    ) -> Result<ExecutionOutput> {
        let mut output_rows = Vec::new();

        for sequence in batch.sequences() {
            let state_index = state_slot_index(sequence.state_slot, states.len())?;
            let state = &mut states[state_index];

            let (query_start, query_end) = query_range(sequence)?;

            match sequence.phase {
                ForwardPhase::Prefill => {
                    let token_ids = &batch.token_ids()[query_start..query_end];
                    let last_row = query_end - 1;
                    let logits_request = batch.logits()[last_row];
                    execute_prefill_sequence(
                        &mut self.runner,
                        state,
                        token_ids,
                        logits_request,
                        last_row,
                        &mut output_rows,
                    )?;
                }
                ForwardPhase::Decode => {
                    let row = query_start;
                    let token_id = batch.token_ids()[row];
                    let logits_request = batch.logits()[row];
                    execute_decode_sequence(
                        &mut self.runner,
                        state,
                        token_id,
                        logits_request,
                        row,
                        &mut output_rows,
                    )?;
                }
            }
        }

        Ok(ExecutionOutput::new(output_rows))
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

fn execute_prefill_sequence<R: MultiSessionRunner>(
    runner: &mut R,
    state: &mut R::SequenceState,
    token_ids: &[u32],
    logits_request: LogitsRequest,
    last_row: usize,
    output_rows: &mut Vec<LogitsRow>,
) -> Result<()> {
    match logits_request {
        LogitsRequest::None => {
            runner.with_sequence_state(state, |r| {
                r.prefill_tokens(token_ids, PrefillMode::Batched)
            })?;
        }
        LogitsRequest::TopK(top_k) => {
            let runner_top_k = top_k_to_usize(top_k)?;
            let logits = runner.with_sequence_state(state, |r| {
                r.prefill_topk(token_ids, runner_top_k, PrefillMode::Batched)
            })?;
            let input_row = row_to_u32(last_row, "prefill input row")?;
            output_rows.push(LogitsRow::new(
                input_row,
                LogitsOutput::TopK(convert_logits(logits)),
            ));
        }
        LogitsRequest::Full => {
            return Err(execution_error(
                "native multi-session prefill does not yet support full logits",
            ));
        }
    }
    Ok(())
}

fn execute_decode_sequence<R: MultiSessionRunner>(
    runner: &mut R,
    state: &mut R::SequenceState,
    token_id: u32,
    logits_request: LogitsRequest,
    row: usize,
    output_rows: &mut Vec<LogitsRow>,
) -> Result<()> {
    match logits_request {
        LogitsRequest::None => {
            runner.with_sequence_state(state, |r| r.feed_token(token_id))?;
        }
        LogitsRequest::TopK(top_k) => {
            let runner_top_k = top_k_to_usize(top_k)?;
            let logits =
                runner.with_sequence_state(state, |r| r.decode_topk(token_id, runner_top_k))?;
            let input_row = row_to_u32(row, "decode input row")?;
            output_rows.push(LogitsRow::new(
                input_row,
                LogitsOutput::TopK(convert_logits(logits)),
            ));
        }
        LogitsRequest::Full => {
            return Err(execution_error(
                "native multi-session decode does not yet support full logits",
            ));
        }
    }
    Ok(())
}

fn state_slot_index(slot: StateSlot, state_count: usize) -> Result<usize> {
    let index = slot.try_as_usize().map_err(|_| {
        execution_error(format!(
            "state slot {} cannot be represented as usize",
            slot.get()
        ))
    })?;
    if index >= state_count {
        return Err(execution_error(format!(
            "state slot {} is out of range for {state_count} states",
            slot.get()
        )));
    }
    Ok(index)
}

fn query_range(sequence: &ferrule_common::execution::ExecutionSequence) -> Result<(usize, usize)> {
    let start = usize::try_from(sequence.query.start)
        .map_err(|_| execution_error("query start overflows usize"))?;
    let end = usize::try_from(sequence.query.end)
        .map_err(|_| execution_error("query end overflows usize"))?;
    Ok((start, end))
}

fn top_k_to_usize(top_k: NonZeroU32) -> Result<usize> {
    usize::try_from(top_k.get()).map_err(|_| {
        execution_error(format!(
            "requested top-k {} cannot be represented as usize",
            top_k.get()
        ))
    })
}

fn row_to_u32(row: usize, context: &str) -> Result<u32> {
    u32::try_from(row).map_err(|_| execution_error(format!("{context} {row} overflows u32")))
}

fn convert_logits(logits: Vec<TokenLogit>) -> Vec<ExecutionTokenLogit> {
    logits
        .into_iter()
        .map(|entry| ExecutionTokenLogit::new(entry.token_id, entry.logit))
        .collect()
}

fn execution_error(message: impl Into<String>) -> Error {
    Error::Execution(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrule_common::execution::{
        ExecutionBatch, ExecutionCapabilities, ExecutionSequence, ForwardMode, LogitsRequest,
    };
    use ferrule_common::{ExpertResidencyControl, ExpertResidencyRequirements};
    use ferrule_model::{
        ModelInfo, ModelRunner, MultiSessionRunner, PrefillMode, TokenLogit, TopKModelRunner,
    };

    /// A mock runner that supports multi-session execution by tracking per-sequence
    /// position and KV state in a simple way.
    #[derive(Default)]
    struct MockMultiSessionRunner {
        position: usize,
        sequences_created: usize,
        sequences_forked: usize,
        transactional: bool,
        prepares: usize,
        commits: usize,
        rollbacks: usize,
        expert_residency_requirements: Option<ExpertResidencyRequirements>,
        expert_residency_install_attempts: usize,
        expert_residency_control: Option<Box<dyn ExpertResidencyControl>>,
        fail_expert_residency_install: bool,
        decode_topk_calls: usize,
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
                .map(|t| char::from_u32(*t).unwrap_or('?'))
                .collect())
        }
        fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
            self.position += tokens.len();
            Ok(vec![0.0, 1.0])
        }
        fn decode_token(&mut self, _token: u32) -> Result<Vec<f32>> {
            self.position += 1;
            Ok(vec![0.0, 1.0])
        }
        fn reset_session(&mut self) -> Result<()> {
            self.position = 0;
            Ok(())
        }
        fn eos_token_id(&self) -> Option<u32> {
            None
        }
    }

    impl TopKModelRunner for MockMultiSessionRunner {
        fn position(&self) -> usize {
            self.position
        }
        fn feed_token(&mut self, _token_id: u32) -> Result<()> {
            if self.expert_residency_requirements.is_some()
                && self.expert_residency_control.is_none()
            {
                return Err(Error::Execution(
                    "mock execution started before expert residency installation".into(),
                ));
            }
            self.position += 1;
            Ok(())
        }
        fn max_top_k(&self) -> usize {
            40
        }
        fn prefill_topk(
            &mut self,
            token_ids: &[u32],
            _top_k: usize,
            _mode: PrefillMode,
        ) -> Result<Vec<TokenLogit>> {
            self.position += token_ids.len();
            Ok(vec![TokenLogit {
                token_id: 1,
                logit: 2.0,
            }])
        }
        fn decode_topk(&mut self, _token_id: u32, _top_k: usize) -> Result<Vec<TokenLogit>> {
            if self.expert_residency_requirements.is_some()
                && self.expert_residency_control.is_none()
            {
                return Err(Error::Execution(
                    "mock execution started before expert residency installation".into(),
                ));
            }
            self.position += 1;
            self.decode_topk_calls += 1;
            Ok(vec![TokenLogit {
                token_id: 2,
                logit: 3.0,
            }])
        }
    }

    /// Per-sequence state for the mock runner.
    #[derive(Debug)]
    struct MockSequenceState {
        position: usize,
        fed_tokens: Vec<u32>,
    }

    impl MultiSessionRunner for MockMultiSessionRunner {
        type SequenceState = MockSequenceState;

        fn expert_residency_requirements(&self) -> Option<ExpertResidencyRequirements> {
            self.expert_residency_requirements.clone()
        }

        fn expert_residency_control_installed(&self) -> bool {
            self.expert_residency_control.is_some()
        }

        fn install_expert_residency_control(
            &mut self,
            control: Box<dyn ExpertResidencyControl>,
        ) -> Result<()> {
            self.expert_residency_install_attempts += 1;
            if self.fail_expert_residency_install {
                return Err(Error::Execution(
                    "mock expert residency install failed".into(),
                ));
            }
            self.expert_residency_control = Some(control);
            Ok(())
        }

        fn prepare_multi_session_batch(
            &mut self,
            _states: &mut [Self::SequenceState],
            _batch: &ExecutionBatch,
            _kv_reservations: &[KvReservation],
        ) -> Result<bool> {
            if self.transactional {
                self.prepares += 1;
            }
            Ok(self.transactional)
        }

        fn commit_multi_session_batch(&mut self) -> Result<()> {
            self.commits += 1;
            Ok(())
        }

        fn rollback_multi_session_batch(&mut self) -> Result<()> {
            self.rollbacks += 1;
            Ok(())
        }

        fn with_sequence_state<T>(
            &mut self,
            state: &mut Self::SequenceState,
            execute: impl FnOnce(&mut Self) -> Result<T>,
        ) -> Result<T> {
            let saved = self.position;
            self.position = state.position;
            let result = execute(self);
            state.position = self.position;
            self.position = saved;
            result
        }

        fn create_sequence_state(&mut self) -> Result<Self::SequenceState> {
            self.sequences_created += 1;
            Ok(MockSequenceState {
                position: 0,
                fed_tokens: Vec::new(),
            })
        }

        fn fork_sequence_state(&mut self) -> Result<Self::SequenceState> {
            self.sequences_forked += 1;
            Ok(MockSequenceState {
                position: 0,
                fed_tokens: Vec::new(),
            })
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
                fed_tokens: source.fed_tokens.clone(),
            })
        }

        fn reset_sequence_state(&mut self, state: &mut Self::SequenceState) -> Result<()> {
            state.position = 0;
            state.fed_tokens.clear();
            Ok(())
        }

        fn release_sequence_state(&mut self, _state: Self::SequenceState) -> Result<()> {
            Ok(())
        }

        fn multi_session_capabilities(&self) -> ExecutionCapabilities {
            ExecutionCapabilities {
                max_batch_tokens: 1024,
                max_sequences: 4,
                max_prefill_query_tokens_per_sequence: 1024,
                max_decode_query_tokens_per_sequence: 1,
                max_top_k: NonZeroU32::new(40),
                supports_prefill: true,
                supports_decode: true,
                supports_mixed: true,
                full_logits_width: None,
                kv_binding_mode: ferrule_common::execution::KvBindingMode::None,
                logits_row_policy: ferrule_common::execution::LogitsRowPolicy::LastPerSequence,
            }
        }
    }

    #[test]
    fn serving_admission_constructs_fresh_state_without_forking_default_session() {
        let mut executor = NativeMultiSessionExecutor::new(MockMultiSessionRunner::default());
        let state = executor.create_sequence_state().unwrap();
        assert_eq!(state.position, 0);
        assert_eq!(executor.runner().sequences_created, 1);
        assert_eq!(executor.runner().sequences_forked, 0);
    }

    fn nz(n: u32) -> NonZeroU32 {
        NonZeroU32::new(n).unwrap()
    }

    fn make_decode_batch(num_sequences: usize) -> (Vec<MockSequenceState>, ExecutionBatch) {
        let states: Vec<_> = (0..num_sequences)
            .map(|_| MockSequenceState {
                position: 10,
                fed_tokens: Vec::new(),
            })
            .collect();

        let mut token_ids = Vec::new();
        let mut positions = Vec::new();
        let mut logits = Vec::new();
        let mut sequences = Vec::new();

        for i in 0..num_sequences {
            let row = i as u32;
            token_ids.push(100 + i as u32);
            positions.push(10);
            logits.push(LogitsRequest::TopK(nz(5)));
            sequences.push(ExecutionSequence::new(
                StateSlot::new(row),
                ForwardPhase::Decode,
                row..row + 1,
                10,
                11,
                0..0,
            ));
        }

        let batch = ExecutionBatch::new(
            ForwardMode::Decode,
            token_ids,
            positions,
            vec![None; num_sequences],
            logits,
            sequences,
            Vec::new(),
        );

        (states, batch)
    }

    #[test]
    fn expert_residency_controller_is_shared_across_batches_and_survives_reset() {
        let requirements = ExpertResidencyRequirements::new(73, vec![2, 3]);
        let runner = MockMultiSessionRunner {
            expert_residency_requirements: Some(requirements.clone()),
            ..MockMultiSessionRunner::default()
        };
        let mut executor = NativeMultiSessionExecutor::new(runner);

        assert_eq!(executor.runner().expert_residency_install_attempts, 0);
        assert!(executor.runner().expert_residency_control.is_none());
        executor.create_sequence_state().unwrap();
        assert_eq!(executor.runner().expert_residency_install_attempts, 0);

        let (mut first_states, first_batch) = make_decode_batch(1);
        executor
            .execute_batch(&mut first_states, &first_batch)
            .unwrap();
        assert_eq!(executor.runner().expert_residency_install_attempts, 1);
        assert_eq!(
            executor
                .runner()
                .expert_residency_control
                .as_deref()
                .unwrap()
                .requirements(),
            requirements
        );
        let first_control = executor
            .runner()
            .expert_residency_control
            .as_deref()
            .unwrap() as *const dyn ExpertResidencyControl as *const ();

        executor.reset().unwrap();
        let (mut second_states, second_batch) = make_decode_batch(1);
        executor
            .execute_batch(&mut second_states, &second_batch)
            .unwrap();
        let second_control = executor
            .runner()
            .expert_residency_control
            .as_deref()
            .unwrap() as *const dyn ExpertResidencyControl
            as *const ();
        assert_eq!(executor.runner().expert_residency_install_attempts, 1);
        assert_eq!(first_control, second_control);

        let runner = executor.into_runner().unwrap();
        assert_eq!(
            runner
                .expert_residency_control
                .as_deref()
                .unwrap()
                .requirements(),
            requirements
        );
        let mut rebuilt = NativeMultiSessionExecutor::new(runner);
        let (mut third_states, third_batch) = make_decode_batch(1);
        rebuilt
            .execute_batch(&mut third_states, &third_batch)
            .unwrap();
        assert_eq!(rebuilt.runner().expert_residency_install_attempts, 1);
        let third_control = rebuilt
            .runner()
            .expert_residency_control
            .as_deref()
            .unwrap() as *const dyn ExpertResidencyControl as *const ();
        assert_eq!(first_control, third_control);
    }

    #[test]
    fn expert_residency_install_failure_poisons_reports_and_can_be_reset() {
        let runner = MockMultiSessionRunner {
            expert_residency_requirements: Some(ExpertResidencyRequirements::new(91, vec![1])),
            fail_expert_residency_install: true,
            ..MockMultiSessionRunner::default()
        };
        let mut executor = NativeMultiSessionExecutor::new(runner);
        let (mut states, batch) = make_decode_batch(1);

        let error = executor.execute_batch(&mut states, &batch).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("mock expert residency install failed")
        );
        assert!(executor.is_poisoned());
        assert_eq!(
            executor.poison_operation(),
            Some("expert_residency_initialization")
        );
        assert!(
            executor
                .poison_cause()
                .unwrap()
                .contains("mock expert residency install failed")
        );
        assert_eq!(executor.runner().expert_residency_install_attempts, 1);
        assert_eq!(executor.runner().decode_topk_calls, 0);

        let poisoned = executor.execute_batch(&mut states, &batch).unwrap_err();
        assert!(poisoned.to_string().contains("native executor is poisoned"));
        assert_eq!(executor.runner().expert_residency_install_attempts, 1);

        executor.runner_mut().fail_expert_residency_install = false;
        executor.reset().unwrap();
        assert!(!executor.is_poisoned());
        executor.execute_batch(&mut states, &batch).unwrap();
        assert_eq!(executor.runner().expert_residency_install_attempts, 2);
        assert_eq!(executor.runner().decode_topk_calls, 1);
    }

    #[test]
    fn expert_residency_is_installed_before_feed_and_diagnostic_paths() {
        let requirements = ExpertResidencyRequirements::new(102, vec![1]);
        let runner = MockMultiSessionRunner {
            expert_residency_requirements: Some(requirements.clone()),
            ..MockMultiSessionRunner::default()
        };
        let mut executor = NativeMultiSessionExecutor::new(runner);
        let mut state = MockSequenceState {
            position: 0,
            fed_tokens: Vec::new(),
        };

        executor.feed_sequence_token(&mut state, 4).unwrap();
        assert_eq!(state.position, 1);
        assert_eq!(executor.runner().expert_residency_install_attempts, 1);

        let (mut states, batch) = make_decode_batch(1);
        executor
            .execute_diagnostic_batch_with_kv(&mut states, &batch, &[], |runner, _states| {
                assert_eq!(
                    runner
                        .expert_residency_control
                        .as_deref()
                        .unwrap()
                        .requirements(),
                    requirements
                );
                Ok(())
            })
            .unwrap();
        assert_eq!(executor.runner().expert_residency_install_attempts, 1);
    }

    #[test]
    fn runner_without_expert_residency_requirements_never_receives_control() {
        let mut executor = NativeMultiSessionExecutor::new(MockMultiSessionRunner::default());
        let (mut states, batch) = make_decode_batch(1);

        executor.execute_batch(&mut states, &batch).unwrap();
        executor.execute_batch(&mut states, &batch).unwrap();

        assert!(executor.expert_residency_initialized);
        assert_eq!(executor.runner().expert_residency_install_attempts, 0);
        assert!(executor.runner().expert_residency_control.is_none());
    }

    #[test]
    fn executor_reports_runner_capabilities() {
        let runner = MockMultiSessionRunner::default();
        let executor = NativeMultiSessionExecutor::new(runner);
        let caps = executor.capabilities();
        assert_eq!(caps.max_sequences, 4);
        assert!(caps.supports_mixed);
        assert!(caps.supports_prefill);
        assert!(caps.supports_decode);
    }

    #[test]
    fn execute_multi_sequence_decode_produces_one_output_per_sequence() {
        let runner = MockMultiSessionRunner::default();
        let mut executor = NativeMultiSessionExecutor::new(runner);
        let (mut states, batch) = make_decode_batch(3);
        let output = executor.execute_batch(&mut states, &batch).unwrap();
        assert_eq!(output.logits.len(), 3);
        assert_eq!(output.logits[0].input_row, 0);
        assert_eq!(output.logits[1].input_row, 1);
        assert_eq!(output.logits[2].input_row, 2);
    }

    #[test]
    fn execute_decode_without_logits_returns_empty_output() {
        let runner = MockMultiSessionRunner::default();
        let mut executor = NativeMultiSessionExecutor::new(runner);

        let mut states = vec![MockSequenceState {
            position: 5,
            fed_tokens: Vec::new(),
        }];

        let batch = ExecutionBatch::new(
            ForwardMode::Decode,
            vec![42],
            vec![5],
            vec![None],
            vec![LogitsRequest::None],
            vec![ExecutionSequence::new(
                StateSlot::new(0),
                ForwardPhase::Decode,
                0..1,
                5,
                6,
                0..0,
            )],
            Vec::new(),
        );

        let output = executor.execute_batch(&mut states, &batch).unwrap();
        assert!(output.logits.is_empty());
    }

    #[test]
    fn execute_prefill_with_topk_returns_last_row_output() {
        let runner = MockMultiSessionRunner::default();
        let mut executor = NativeMultiSessionExecutor::new(runner);

        let mut states = vec![MockSequenceState {
            position: 0,
            fed_tokens: Vec::new(),
        }];

        let batch = ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![1, 2, 3],
            vec![0, 1, 2],
            vec![None, None, None],
            vec![
                LogitsRequest::None,
                LogitsRequest::None,
                LogitsRequest::TopK(nz(5)),
            ],
            vec![ExecutionSequence::new(
                StateSlot::new(0),
                ForwardPhase::Prefill,
                0..3,
                0,
                3,
                0..0,
            )],
            Vec::new(),
        );

        let output = executor.execute_batch(&mut states, &batch).unwrap();
        assert_eq!(output.logits.len(), 1);
        assert_eq!(output.logits[0].input_row, 2);
    }

    #[test]
    fn execute_mixed_batch_routes_prefill_and_decode_separately() {
        let runner = MockMultiSessionRunner::default();
        let mut executor = NativeMultiSessionExecutor::new(runner);

        // Sequence 0: prefill 2 tokens, last row gets top-k
        // Sequence 1: decode 1 token with top-k
        let mut states = vec![
            MockSequenceState {
                position: 0,
                fed_tokens: Vec::new(),
            },
            MockSequenceState {
                position: 5,
                fed_tokens: Vec::new(),
            },
        ];

        let batch = ExecutionBatch::new(
            ForwardMode::Mixed,
            vec![10, 20, 30],
            vec![0, 1, 5],
            vec![None, None, None],
            vec![
                LogitsRequest::None,
                LogitsRequest::TopK(nz(5)),
                LogitsRequest::TopK(nz(5)),
            ],
            vec![
                ExecutionSequence::new(StateSlot::new(0), ForwardPhase::Prefill, 0..2, 0, 2, 0..0),
                ExecutionSequence::new(StateSlot::new(1), ForwardPhase::Decode, 2..3, 5, 6, 0..0),
            ],
            Vec::new(),
        );

        let output = executor.execute_batch(&mut states, &batch).unwrap();
        assert_eq!(output.logits.len(), 2);
        assert_eq!(output.logits[0].input_row, 1); // prefill last row
        assert_eq!(output.logits[1].input_row, 2); // decode row
    }

    #[test]
    fn batched_decode_matches_serial_execution_exactly() {
        let (mut batched_states, batch) = make_decode_batch(2);
        let mut batched = NativeMultiSessionExecutor::new(MockMultiSessionRunner::default());
        let batched_output = batched.execute_batch(&mut batched_states, &batch).unwrap();

        let mut serial = NativeMultiSessionExecutor::new(MockMultiSessionRunner::default());
        let (mut state_a, batch_a) = make_decode_batch(1);
        let output_a = serial.execute_batch(&mut state_a, &batch_a).unwrap();
        let (mut state_b, batch_b) = make_decode_batch(1);
        let output_b = serial.execute_batch(&mut state_b, &batch_b).unwrap();

        assert_eq!(batched_output.logits[0].logits, output_a.logits[0].logits);
        assert_eq!(batched_output.logits[1].logits, output_b.logits[0].logits);
        assert_eq!(batched_states[0].position, state_a[0].position);
        assert_eq!(batched_states[1].position, state_b[0].position);
    }

    #[test]
    fn backend_batch_transaction_requires_explicit_commit_or_rollback() {
        let runner = MockMultiSessionRunner {
            transactional: true,
            ..MockMultiSessionRunner::default()
        };
        let mut executor = NativeMultiSessionExecutor::new(runner);
        let (mut states, batch) = make_decode_batch(1);

        executor.execute_batch(&mut states, &batch).unwrap();
        assert_eq!(executor.runner().prepares, 1);
        assert!(executor.execute_batch(&mut states, &batch).is_err());
        executor.commit_prepared_batch().unwrap();
        assert_eq!(executor.runner().commits, 1);

        let (_, next_batch) = make_decode_batch(1);
        executor.execute_batch(&mut states, &next_batch).unwrap();
        executor.rollback_prepared_batch().unwrap();
        assert_eq!(executor.runner().rollbacks, 1);
    }

    #[test]
    fn reset_of_one_sequence_does_not_change_another() {
        let mut executor = NativeMultiSessionExecutor::new(MockMultiSessionRunner::default());
        let (mut states, batch) = make_decode_batch(2);
        executor.execute_batch(&mut states, &batch).unwrap();
        let b_position = states[1].position;
        executor.reset_sequence_state(&mut states[0]).unwrap();
        assert_eq!(states[0].position, 0);
        assert_eq!(states[1].position, b_position);
    }

    #[test]
    fn poisoned_executor_rejects_further_execution() {
        let runner = MockMultiSessionRunner::default();
        let mut executor = NativeMultiSessionExecutor::new(runner);
        executor.poison = Some(PoisonState {
            operation: "test",
            cause: "test failure".into(),
        });
        assert!(executor.is_poisoned());
        let (mut states, batch) = make_decode_batch(1);
        let err = executor.execute_batch(&mut states, &batch).unwrap_err();
        assert!(err.to_string().contains("poisoned"));
    }

    #[test]
    fn reset_clears_poison() {
        let runner = MockMultiSessionRunner::default();
        let mut executor = NativeMultiSessionExecutor::new(runner);
        executor.poison = Some(PoisonState {
            operation: "test",
            cause: "test failure".into(),
        });
        assert!(executor.is_poisoned());
        executor.reset().unwrap();
        assert!(!executor.is_poisoned());
    }

    #[test]
    fn state_slot_index_rejects_out_of_range() {
        assert!(state_slot_index(StateSlot::new(0), 2).is_ok());
        assert!(state_slot_index(StateSlot::new(1), 2).is_ok());
        assert!(state_slot_index(StateSlot::new(2), 2).is_err());
    }

    #[test]
    fn convert_logits_preserves_order_and_values() {
        let input = vec![
            TokenLogit {
                token_id: 1,
                logit: 2.0,
            },
            TokenLogit {
                token_id: 5,
                logit: 1.0,
            },
        ];
        let output = convert_logits(input);
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].token_id, 1);
        assert_eq!(output[0].logit, 2.0);
        assert_eq!(output[1].token_id, 5);
        assert_eq!(output[1].logit, 1.0);
    }
}
