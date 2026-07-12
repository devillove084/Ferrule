use std::num::NonZeroU32;

use ferrule_common::execution::{
    ExecutionBatch, ExecutionCapabilities, ExecutionOutput, ForwardMode, KvBindingMode,
    LogitsOutput, LogitsRequest, LogitsRow, LogitsRowPolicy, TokenLogit as ExecutionTokenLogit,
};
use ferrule_common::{Error, Result};
use ferrule_model::{PrefillMode, TopKModelRunner};

/// Execution policy for the synchronous resident top-k compatibility adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopKCompatibilityExecutorConfig {
    /// Default candidate count exposed to runtime lowering through `default_top_k`.
    /// A neutral execution batch's per-row request remains authoritative at execution.
    pub top_k: usize,
    /// Prompt execution strategy used for prefill chunks.
    pub prefill_mode: PrefillMode,
}

impl Default for TopKCompatibilityExecutorConfig {
    fn default() -> Self {
        Self {
            top_k: 1,
            prefill_mode: PrefillMode::Interactive,
        }
    }
}

/// Explicit single-sequence compatibility adapter from the neutral execution ABI
/// to a stateful `TopKModelRunner`.
///
/// A plain `TopKModelRunner` owns one implicit resident sequence and has no physical
/// paged-KV binding surface. The advertised capabilities make those limitations
/// visible before execution rather than discovering them after runner mutation.
pub struct TopKCompatibilityExecutor<R: TopKModelRunner> {
    runner: R,
    config: TopKCompatibilityExecutorConfig,
    poison: Option<TopKCompatibilityExecutorPoison>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TopKCompatibilityExecutorPoison {
    operation: &'static str,
    cause: String,
}

impl<R: TopKModelRunner> TopKCompatibilityExecutor<R> {
    pub fn new(runner: R) -> Self {
        Self::with_config(runner, TopKCompatibilityExecutorConfig::default())
    }

    pub fn with_config(runner: R, config: TopKCompatibilityExecutorConfig) -> Self {
        Self {
            runner,
            config,
            poison: None,
        }
    }

    /// Returns the configured top-k default in the neutral ABI's checked wire type.
    pub fn default_top_k(&self) -> Result<NonZeroU32> {
        let top_k = u32::try_from(self.config.top_k).map_err(|_| {
            execution_error(format!(
                "configured top_k {} exceeds the neutral ABI u32 range",
                self.config.top_k
            ))
        })?;
        NonZeroU32::new(top_k)
            .ok_or_else(|| execution_error("configured top_k must be greater than zero"))
    }

    /// Truthful limits of the stateful single-sequence compatibility path.
    pub fn capabilities(&self) -> ExecutionCapabilities {
        let max_top_k = u32::try_from(self.runner.max_top_k()).unwrap_or(u32::MAX);
        ExecutionCapabilities {
            max_batch_tokens: usize::MAX,
            max_sequences: 1,
            max_prefill_query_tokens_per_sequence: usize::MAX,
            max_decode_query_tokens_per_sequence: 1,
            max_top_k: NonZeroU32::new(max_top_k),
            supports_prefill: true,
            supports_decode: true,
            supports_mixed: false,
            full_logits_width: None,
            kv_binding_mode: KvBindingMode::None,
            logits_row_policy: LogitsRowPolicy::LastPerSequence,
        }
    }

    pub fn runner(&self) -> &R {
        &self.runner
    }

    pub fn into_runner(self) -> Result<R> {
        if let Some(poison) = self.poison {
            return Err(Error::Execution(format!(
                "cannot extract resident runner after {} failed: {}; reset or release it instead",
                poison.operation, poison.cause
            )));
        }
        Ok(self.runner)
    }

    pub fn position(&self) -> usize {
        self.runner.position()
    }

    /// Whether a model mutation failed after execution may have partially committed
    /// runner-owned KV or sequence state.
    pub fn is_poisoned(&self) -> bool {
        self.poison.is_some()
    }

    pub fn poison_operation(&self) -> Option<&'static str> {
        self.poison.as_ref().map(|poison| poison.operation)
    }

    pub fn poison_cause(&self) -> Option<&str> {
        self.poison.as_ref().map(|poison| poison.cause.as_str())
    }

    /// Reset the implicit runner session and clear poison only after reset succeeds.
    /// A failed reset leaves the executor poisoned because reset itself may have
    /// partially changed runner-owned state.
    pub fn reset_session(&mut self) -> Result<()> {
        match self.runner.reset_session() {
            Ok(()) => {
                self.poison = None;
                Ok(())
            }
            Err(error) => {
                self.record_poison("reset_session", &error);
                Err(error)
            }
        }
    }

    /// Append one token through the same poison-aware mutation boundary used by
    /// batched execution. This is used for terminal EOS state updates.
    pub fn feed_token(&mut self, token_id: u32) -> Result<()> {
        self.run_model_mutation("feed_token", |runner| runner.feed_token(token_id))
    }

    /// Mark the implicit runner state unsafe when runtime correlation validation
    /// fails after a successful model mutation.
    pub(crate) fn poison_after_output_contract(&mut self, error: &Error) {
        self.record_poison("output_contract", error);
    }

    pub fn execute_batch(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput> {
        self.validate_batch_for_adapter(batch)?;

        match batch.mode() {
            ForwardMode::Prefill => self.execute_prefill_batch(batch),
            ForwardMode::Decode => self.execute_decode_batch(batch),
            ForwardMode::Mixed => Err(execution_error(
                "resident top-k compatibility adapter does not support mixed execution batches",
            )),
        }
    }

    /// Runs every generic and adapter-specific shape/state check before a runner
    /// mutation is attempted. Errors from this phase never poison the executor.
    fn validate_batch_for_adapter(&self, batch: &ExecutionBatch) -> Result<()> {
        batch.validate(1, &self.capabilities())?;

        let [sequence] = batch.sequences() else {
            return Err(execution_error(format!(
                "resident top-k compatibility adapter requires exactly one sequence, got {}",
                batch.sequences().len()
            )));
        };
        if sequence.state_slot.get() != 0 {
            return Err(execution_error(format!(
                "resident top-k compatibility adapter requires state slot 0, got {}",
                sequence.state_slot.get()
            )));
        }

        match batch.mode() {
            ForwardMode::Prefill => {
                let last_row = batch.len() - 1;
                if let Some(row) = batch.logits()[..last_row]
                    .iter()
                    .position(|request| !matches!(request, LogitsRequest::None))
                {
                    return Err(execution_error(format!(
                        "resident top-k prefill can request logits only on its final input row, but row {row} requested logits"
                    )));
                }
                if matches!(batch.logits()[last_row], LogitsRequest::Full) {
                    return Err(execution_error(
                        "resident top-k prefill does not support full logits",
                    ));
                }
            }
            ForwardMode::Decode => {
                if batch.len() != 1 {
                    return Err(execution_error(format!(
                        "resident top-k decode requires exactly one input row, got {}",
                        batch.len()
                    )));
                }
                if matches!(batch.logits()[0], LogitsRequest::Full) {
                    return Err(execution_error(
                        "resident top-k decode does not support full logits",
                    ));
                }
            }
            ForwardMode::Mixed => {
                return Err(execution_error(
                    "resident top-k compatibility adapter does not support mixed execution batches",
                ));
            }
        }

        let runner_position = self.runner.position();
        let runner_position_u32 = u32::try_from(runner_position).map_err(|_| {
            execution_error(format!(
                "resident runner position {runner_position} exceeds the neutral ABI u32 range"
            ))
        })?;
        if sequence.context_len != runner_position_u32 {
            return Err(execution_error(format!(
                "{:?} context mismatch: runner at {runner_position}, batch context is {}",
                batch.mode(),
                sequence.context_len
            )));
        }

        Ok(())
    }

    fn execute_prefill_batch(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput> {
        let last_row = batch.len() - 1;
        match batch.logits()[last_row] {
            LogitsRequest::None => {
                let prefill_mode = self.config.prefill_mode;
                self.run_model_mutation("prefill_tokens", |runner| {
                    runner.prefill_tokens(batch.token_ids(), prefill_mode)
                })?;
                self.validate_output_after_mutation(
                    "prefill_tokens",
                    batch,
                    ExecutionOutput::default(),
                )
            }
            LogitsRequest::TopK(top_k) => {
                let runner_top_k = Self::runner_top_k(top_k)?;
                let input_row = u32::try_from(last_row).map_err(|_| {
                    execution_error(format!(
                        "prefill input row {last_row} exceeds the neutral ABI u32 range"
                    ))
                })?;
                let prefill_mode = self.config.prefill_mode;
                let logits = self.run_model_mutation("prefill_topk", |runner| {
                    runner.prefill_topk(batch.token_ids(), runner_top_k, prefill_mode)
                })?;
                let output = ExecutionOutput::new(vec![LogitsRow::new(
                    input_row,
                    LogitsOutput::TopK(Self::neutral_top_k(logits)),
                )]);
                self.validate_output_after_mutation("prefill_topk", batch, output)
            }
            LogitsRequest::Full => Err(execution_error(
                "resident top-k prefill does not support full logits",
            )),
        }
    }

    fn execute_decode_batch(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput> {
        let token_id = batch.token_ids()[0];
        match batch.logits()[0] {
            LogitsRequest::None => {
                self.run_model_mutation("feed_token", |runner| runner.feed_token(token_id))?;
                self.validate_output_after_mutation("feed_token", batch, ExecutionOutput::default())
            }
            LogitsRequest::TopK(top_k) => {
                let runner_top_k = Self::runner_top_k(top_k)?;
                let logits = self.run_model_mutation("decode_topk", |runner| {
                    runner.decode_topk(token_id, runner_top_k)
                })?;
                let output = ExecutionOutput::new(vec![LogitsRow::new(
                    0,
                    LogitsOutput::TopK(Self::neutral_top_k(logits)),
                )]);
                self.validate_output_after_mutation("decode_topk", batch, output)
            }
            LogitsRequest::Full => Err(execution_error(
                "resident top-k decode does not support full logits",
            )),
        }
    }

    fn runner_top_k(top_k: NonZeroU32) -> Result<usize> {
        usize::try_from(top_k.get()).map_err(|_| {
            execution_error(format!(
                "requested top-k {} cannot be represented by the runner's usize API",
                top_k.get()
            ))
        })
    }

    fn neutral_top_k(logits: Vec<ferrule_model::TokenLogit>) -> Vec<ExecutionTokenLogit> {
        logits
            .into_iter()
            .map(|entry| ExecutionTokenLogit::new(entry.token_id, entry.logit))
            .collect()
    }

    /// Output validation happens after a successful runner call, so any contract
    /// failure means runner state advanced without a trustworthy corresponding
    /// output. Record poison before returning that validation error.
    fn validate_output_after_mutation(
        &mut self,
        operation: &'static str,
        batch: &ExecutionBatch,
        output: ExecutionOutput,
    ) -> Result<ExecutionOutput> {
        if let Err(error) = output.validate_with_capabilities(batch, &self.capabilities()) {
            self.record_poison(operation, &error);
            return Err(error);
        }
        Ok(output)
    }

    fn run_model_mutation<T>(
        &mut self,
        operation: &'static str,
        mutation: impl FnOnce(&mut R) -> Result<T>,
    ) -> Result<T> {
        self.ensure_ready()?;
        match mutation(&mut self.runner) {
            Ok(value) => Ok(value),
            Err(error) => {
                self.record_poison(operation, &error);
                Err(error)
            }
        }
    }

    fn ensure_ready(&self) -> Result<()> {
        let Some(poison) = &self.poison else {
            return Ok(());
        };
        Err(Error::Internal(format!(
            "resident executor is poisoned after {} failed: {}; reset or release the runner before executing again",
            poison.operation, poison.cause
        )))
    }

    fn record_poison(&mut self, operation: &'static str, error: &Error) {
        self.poison = Some(TopKCompatibilityExecutorPoison {
            operation,
            cause: error.to_string(),
        });
    }
}

fn execution_error(message: impl Into<String>) -> Error {
    Error::Execution(message.into())
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::ops::Range;

    use ferrule_common::execution::{
        ExecutionSequence, ForwardPhase, KvWriteSlot, StateSlot, TokenLogit as NeutralTokenLogit,
    };
    use ferrule_common::{Error, Result};
    use ferrule_model::{ModelInfo, ModelRunner, TokenLogit};

    use super::*;

    #[derive(Debug)]
    struct MockTopKRunner {
        position: usize,
        outputs: VecDeque<Vec<TokenLogit>>,
        fed: Vec<u32>,
        prefill_tokens_calls: Vec<Vec<u32>>,
        prefill_topk_calls: Vec<Vec<u32>>,
        requested_top_k: Vec<usize>,
        fail_next_mutation: bool,
        fail_next_reset: bool,
        mutation_calls: usize,
        max_top_k: usize,
    }

    impl MockTopKRunner {
        fn new(outputs: Vec<Vec<TokenLogit>>) -> Self {
            Self {
                position: 0,
                outputs: outputs.into(),
                fed: Vec::new(),
                prefill_tokens_calls: Vec::new(),
                prefill_topk_calls: Vec::new(),
                requested_top_k: Vec::new(),
                fail_next_mutation: false,
                fail_next_reset: false,
                mutation_calls: 0,
                max_top_k: usize::MAX,
            }
        }

        fn failing_next_mutation(mut self) -> Self {
            self.fail_next_mutation = true;
            self
        }

        fn failing_next_reset(mut self) -> Self {
            self.fail_next_reset = true;
            self
        }

        fn with_max_top_k(mut self, max_top_k: usize) -> Self {
            self.max_top_k = max_top_k;
            self
        }

        fn complete_mutation<T>(&mut self, value: T) -> Result<T> {
            self.mutation_calls += 1;
            if std::mem::take(&mut self.fail_next_mutation) {
                Err(Error::Model(
                    "simulated failure after partial runner mutation".into(),
                ))
            } else {
                Ok(value)
            }
        }

        fn next_output(&mut self) -> Vec<TokenLogit> {
            self.outputs.pop_front().unwrap_or_else(|| {
                vec![TokenLogit {
                    token_id: 0,
                    logit: 0.0,
                }]
            })
        }
    }

    impl ModelRunner for MockTopKRunner {
        fn model_info(&self) -> ModelInfo {
            ModelInfo {
                family: ferrule_model::ModelFamily::Unknown("mock".into()),
                architecture: Some("mock".into()),
                attention: ferrule_model::AttentionKind::Unknown("mock".into()),
                weight_source: ferrule_model::WeightSource::Unknown,
                hidden_size: 1,
                num_layers: 1,
                num_experts: 0,
                num_experts_per_tok: 0,
                vocab_size: 128,
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

        fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
            self.position += tokens.len();
            Ok(vec![0.0])
        }

        fn decode_token(&mut self, token: u32) -> Result<Vec<f32>> {
            self.fed.push(token);
            self.position += 1;
            Ok(vec![0.0])
        }

        fn reset_session(&mut self) -> Result<()> {
            self.position = 0;
            self.fed.clear();
            self.prefill_tokens_calls.clear();
            self.prefill_topk_calls.clear();
            if std::mem::take(&mut self.fail_next_reset) {
                Err(Error::Model(
                    "simulated failure after partial runner reset".into(),
                ))
            } else {
                Ok(())
            }
        }

        fn eos_token_id(&self) -> Option<u32> {
            None
        }
    }

    impl TopKModelRunner for MockTopKRunner {
        fn position(&self) -> usize {
            self.position
        }

        fn feed_token(&mut self, token_id: u32) -> Result<()> {
            self.fed.push(token_id);
            self.position += 1;
            self.complete_mutation(())
        }

        fn max_top_k(&self) -> usize {
            self.max_top_k
        }

        fn prefill_tokens(&mut self, token_ids: &[u32], _mode: PrefillMode) -> Result<()> {
            self.prefill_tokens_calls.push(token_ids.to_vec());
            self.fed.extend_from_slice(token_ids);
            self.position += token_ids.len();
            self.complete_mutation(())
        }

        fn prefill_topk(
            &mut self,
            token_ids: &[u32],
            top_k: usize,
            _mode: PrefillMode,
        ) -> Result<Vec<TokenLogit>> {
            self.requested_top_k.push(top_k);
            self.prefill_topk_calls.push(token_ids.to_vec());
            self.position += token_ids.len();
            self.complete_mutation(())?;
            Ok(self.next_output())
        }

        fn decode_topk(&mut self, token_id: u32, top_k: usize) -> Result<Vec<TokenLogit>> {
            self.requested_top_k.push(top_k);
            self.feed_token(token_id)?;
            Ok(self.next_output())
        }
    }

    fn nz(value: u32) -> NonZeroU32 {
        NonZeroU32::new(value).unwrap()
    }

    fn model_top(token_id: u32) -> Vec<TokenLogit> {
        vec![TokenLogit {
            token_id,
            logit: 1.0,
        }]
    }

    fn neutral_top(token_id: u32) -> Vec<NeutralTokenLogit> {
        vec![NeutralTokenLogit::new(token_id, 1.0)]
    }

    fn sequence(
        state_slot: u32,
        phase: ForwardPhase,
        query: Range<u32>,
        context_len: u32,
    ) -> ExecutionSequence {
        let query_len = query.end - query.start;
        ExecutionSequence::new(
            StateSlot::new(state_slot),
            phase,
            query,
            context_len,
            context_len.checked_add(query_len).unwrap(),
            0..0,
        )
    }

    fn prefill_batch(
        context_len: u32,
        token_ids: &[u32],
        last_logits: LogitsRequest,
    ) -> ExecutionBatch {
        assert!(!token_ids.is_empty());
        let query_len = u32::try_from(token_ids.len()).unwrap();
        let sequence_len = context_len.checked_add(query_len).unwrap();
        let mut logits = vec![LogitsRequest::None; token_ids.len()];
        logits[token_ids.len() - 1] = last_logits;
        ExecutionBatch::new(
            ForwardMode::Prefill,
            token_ids.to_vec(),
            (context_len..sequence_len).collect(),
            vec![None; token_ids.len()],
            logits,
            vec![sequence(
                0,
                ForwardPhase::Prefill,
                0..query_len,
                context_len,
            )],
            Vec::new(),
        )
    }

    fn decode_batch(context_len: u32, token_id: u32, logits: LogitsRequest) -> ExecutionBatch {
        ExecutionBatch::new(
            ForwardMode::Decode,
            vec![token_id],
            vec![context_len],
            vec![None],
            vec![logits],
            vec![sequence(0, ForwardPhase::Decode, 0..1, context_len)],
            Vec::new(),
        )
    }

    fn multi_sequence_decode_batch() -> ExecutionBatch {
        ExecutionBatch::new(
            ForwardMode::Decode,
            vec![10, 20],
            vec![0, 0],
            vec![None, None],
            vec![LogitsRequest::None, LogitsRequest::None],
            vec![
                sequence(0, ForwardPhase::Decode, 0..1, 0),
                sequence(1, ForwardPhase::Decode, 1..2, 0),
            ],
            Vec::new(),
        )
    }

    fn mixed_batch() -> ExecutionBatch {
        ExecutionBatch::new(
            ForwardMode::Mixed,
            vec![10],
            vec![0],
            vec![None],
            vec![LogitsRequest::None],
            vec![sequence(0, ForwardPhase::Prefill, 0..1, 0)],
            Vec::new(),
        )
    }

    fn physical_kv_decode_batch() -> ExecutionBatch {
        ExecutionBatch::new(
            ForwardMode::Decode,
            vec![10],
            vec![0],
            vec![Some(KvWriteSlot::new(7))],
            vec![LogitsRequest::None],
            vec![sequence(0, ForwardPhase::Decode, 0..1, 0)],
            Vec::new(),
        )
    }

    #[test]
    fn executor_exposes_checked_default_top_k_and_truthful_capabilities() {
        let executor = TopKCompatibilityExecutor::with_config(
            MockTopKRunner::new(Vec::new()),
            TopKCompatibilityExecutorConfig {
                top_k: 7,
                prefill_mode: PrefillMode::Interactive,
            },
        );
        assert_eq!(executor.default_top_k().unwrap(), nz(7));

        let capabilities = executor.capabilities();
        assert_eq!(capabilities.max_batch_tokens, usize::MAX);
        assert_eq!(capabilities.max_sequences, 1);
        assert_eq!(
            capabilities.max_prefill_query_tokens_per_sequence,
            usize::MAX
        );
        assert_eq!(capabilities.max_decode_query_tokens_per_sequence, 1);
        assert_eq!(capabilities.max_top_k, Some(nz(u32::MAX)));
        assert!(capabilities.supports_prefill);
        assert!(capabilities.supports_decode);
        assert!(!capabilities.supports_mixed);
        assert_eq!(capabilities.full_logits_width, None);
        assert_eq!(capabilities.kv_binding_mode, KvBindingMode::None);
        assert_eq!(
            capabilities.logits_row_policy,
            LogitsRowPolicy::LastPerSequence
        );

        let zero = TopKCompatibilityExecutor::with_config(
            MockTopKRunner::new(Vec::new()),
            TopKCompatibilityExecutorConfig {
                top_k: 0,
                prefill_mode: PrefillMode::Interactive,
            },
        );
        assert!(matches!(zero.default_top_k(), Err(Error::Execution(_))));

        let limited =
            TopKCompatibilityExecutor::new(MockTopKRunner::new(Vec::new()).with_max_top_k(40));
        assert_eq!(limited.capabilities().max_top_k, Some(nz(40)));
        let unsupported = decode_batch(0, 7, LogitsRequest::TopK(nz(41)));
        let mut limited = limited;
        assert!(matches!(
            limited.execute_batch(&unsupported),
            Err(Error::Execution(_))
        ));
        assert_eq!(limited.runner().mutation_calls, 0);

        #[cfg(target_pointer_width = "64")]
        {
            let too_large = TopKCompatibilityExecutor::with_config(
                MockTopKRunner::new(Vec::new()),
                TopKCompatibilityExecutorConfig {
                    top_k: u32::MAX as usize + 1,
                    prefill_mode: PrefillMode::Interactive,
                },
            );
            assert!(matches!(
                too_large.default_top_k(),
                Err(Error::Execution(_))
            ));
        }
    }

    #[test]
    fn executor_prefill_returns_last_input_row_and_uses_batch_requested_top_k() {
        let runner = MockTopKRunner::new(vec![model_top(10)]);
        let mut executor = TopKCompatibilityExecutor::with_config(
            runner,
            TopKCompatibilityExecutorConfig {
                top_k: 1,
                prefill_mode: PrefillMode::Interactive,
            },
        );
        let batch = prefill_batch(0, &[1, 2, 3], LogitsRequest::TopK(nz(7)));

        let output = executor.execute_batch(&batch).unwrap();

        assert_eq!(executor.position(), 3);
        assert_eq!(
            output.logits,
            vec![LogitsRow::new(2, LogitsOutput::TopK(neutral_top(10)))]
        );
        assert_eq!(executor.runner().requested_top_k, vec![7]);
        assert_eq!(executor.runner().prefill_topk_calls, vec![vec![1, 2, 3]]);
        assert!(executor.runner().prefill_tokens_calls.is_empty());
    }

    #[test]
    fn executor_prefill_without_logits_returns_empty_output() {
        let runner = MockTopKRunner::new(vec![model_top(99)]);
        let mut executor = TopKCompatibilityExecutor::new(runner);
        let batch = prefill_batch(0, &[1, 2], LogitsRequest::None);

        let output = executor.execute_batch(&batch).unwrap();

        assert_eq!(output, ExecutionOutput::default());
        assert_eq!(executor.position(), 2);
        assert_eq!(executor.runner().prefill_tokens_calls, vec![vec![1, 2]]);
        assert!(executor.runner().prefill_topk_calls.is_empty());
        assert!(executor.runner().requested_top_k.is_empty());
        assert_eq!(executor.runner().fed, vec![1, 2]);
        assert_eq!(executor.runner().outputs.len(), 1);
    }

    #[test]
    fn executor_decode_uses_batch_top_k_and_none_returns_empty_output() {
        let runner = MockTopKRunner::new(vec![model_top(42)]);
        let mut executor = TopKCompatibilityExecutor::with_config(
            runner,
            TopKCompatibilityExecutorConfig {
                top_k: 1,
                prefill_mode: PrefillMode::Interactive,
            },
        );

        let top_k_batch = decode_batch(0, 7, LogitsRequest::TopK(nz(9)));
        let output = executor.execute_batch(&top_k_batch).unwrap();
        assert_eq!(
            output.logits,
            vec![LogitsRow::new(0, LogitsOutput::TopK(neutral_top(42)))]
        );
        assert_eq!(executor.runner().requested_top_k, vec![9]);

        let no_logits_batch = decode_batch(1, 8, LogitsRequest::None);
        let output = executor.execute_batch(&no_logits_batch).unwrap();
        assert_eq!(output, ExecutionOutput::default());
        assert_eq!(executor.position(), 2);
        assert_eq!(executor.runner().fed, vec![7, 8]);
        assert_eq!(executor.runner().mutation_calls, 2);
    }

    #[test]
    fn executor_rejects_unsupported_or_mismatched_batches_before_mutation() {
        let runner = MockTopKRunner::new(Vec::new());
        let mut executor = TopKCompatibilityExecutor::new(runner);
        let invalid_batches = [
            ("multi-sequence", multi_sequence_decode_batch()),
            ("mixed", mixed_batch()),
            ("full logits", decode_batch(0, 10, LogitsRequest::Full)),
            ("physical KV", physical_kv_decode_batch()),
            (
                "position mismatch",
                decode_batch(1, 10, LogitsRequest::None),
            ),
        ];

        for (label, batch) in invalid_batches {
            let error = executor.execute_batch(&batch).unwrap_err();
            assert!(
                matches!(&error, Error::Execution(_)),
                "{label} returned unexpected error: {error}"
            );
            assert_eq!(
                executor.runner().mutation_calls,
                0,
                "{label} reached the runner"
            );
            assert_eq!(executor.position(), 0, "{label} changed runner position");
            assert!(!executor.is_poisoned(), "{label} poisoned the executor");
        }

        let valid = decode_batch(0, 10, LogitsRequest::None);
        assert_eq!(
            executor.execute_batch(&valid).unwrap(),
            ExecutionOutput::default()
        );
        assert_eq!(executor.runner().mutation_calls, 1);
    }

    #[test]
    fn executor_poison_blocks_retry_after_partial_decode_failure_until_reset() {
        let runner = MockTopKRunner::new(vec![model_top(10)]).failing_next_mutation();
        let mut executor = TopKCompatibilityExecutor::new(runner);
        let batch = decode_batch(0, 7, LogitsRequest::TopK(nz(3)));

        let error = executor.execute_batch(&batch).unwrap_err();
        assert!(format!("{error}").contains("simulated failure"));
        assert!(executor.is_poisoned());
        assert_eq!(executor.poison_operation(), Some("decode_topk"));
        assert_eq!(executor.position(), 1, "mock failure partially committed");
        assert_eq!(executor.runner().mutation_calls, 1);
        assert_eq!(executor.runner().requested_top_k, vec![3]);

        let retry = decode_batch(1, 8, LogitsRequest::TopK(nz(3)));
        let error = executor.execute_batch(&retry).unwrap_err();
        assert!(format!("{error}").contains("resident executor is poisoned"));
        assert_eq!(executor.runner().mutation_calls, 1);
        assert_eq!(executor.runner().fed, vec![7]);
        assert_eq!(executor.runner().requested_top_k, vec![3]);

        executor.reset_session().unwrap();
        assert!(!executor.is_poisoned());
        assert_eq!(executor.position(), 0);
        let output = executor.execute_batch(&batch).unwrap();
        assert_eq!(
            output.logits,
            vec![LogitsRow::new(0, LogitsOutput::TopK(neutral_top(10)))]
        );
        assert_eq!(executor.position(), 1);
        assert_eq!(executor.runner().mutation_calls, 2);
        assert_eq!(executor.runner().requested_top_k, vec![3, 3]);
    }

    #[test]
    fn failed_reset_keeps_executor_poisoned() {
        let runner = MockTopKRunner::new(vec![model_top(10)])
            .failing_next_mutation()
            .failing_next_reset();
        let mut executor = TopKCompatibilityExecutor::new(runner);
        let batch = decode_batch(0, 7, LogitsRequest::TopK(nz(1)));
        executor.execute_batch(&batch).unwrap_err();

        let reset_error = executor.reset_session().unwrap_err();
        assert!(format!("{reset_error}").contains("partial runner reset"));
        assert!(executor.is_poisoned());
        assert_eq!(executor.poison_operation(), Some("reset_session"));

        let error = executor.execute_batch(&batch).unwrap_err();
        assert!(format!("{error}").contains("resident executor is poisoned"));
        assert_eq!(executor.runner().mutation_calls, 1);

        executor.reset_session().unwrap();
        assert!(!executor.is_poisoned());
    }

    #[test]
    fn output_contract_failure_after_mutation_poisons_executor() {
        let mut executor = TopKCompatibilityExecutor::new(MockTopKRunner::new(Vec::new()));
        let batch = decode_batch(0, 7, LogitsRequest::None);
        executor.feed_token(7).unwrap();
        let invalid_output =
            ExecutionOutput::new(vec![LogitsRow::new(0, LogitsOutput::TopK(Vec::new()))]);

        let error = executor
            .validate_output_after_mutation("feed_token", &batch, invalid_output)
            .unwrap_err();

        assert!(matches!(error, Error::Execution(_)));
        assert!(executor.is_poisoned());
        assert_eq!(executor.poison_operation(), Some("feed_token"));
        assert_eq!(executor.runner().mutation_calls, 1);
    }
}
