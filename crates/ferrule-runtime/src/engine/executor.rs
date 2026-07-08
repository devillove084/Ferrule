use ferrule_common::{Error, Result};
use ferrule_model::{PrefillMode, TopKModelRunner};

use crate::graph::runtime::{
    ExecutionBatch, ExecutionOutput, ExecutionRowOutput, ExecutionSegment, RowLogits,
};
use crate::scheduling::SchedulerAction;

/// Execution policy for the synchronous resident top-k action executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentActionExecutorConfig {
    /// Number of logits candidates requested from the runner.
    pub top_k: usize,
    /// Prompt execution strategy used for prefill chunks.
    pub prefill_mode: PrefillMode,
}

impl Default for ResidentActionExecutorConfig {
    fn default() -> Self {
        Self {
            top_k: 1,
            prefill_mode: PrefillMode::Interactive,
        }
    }
}

/// Synchronous bridge from scheduler actions to a `TopKModelRunner`.
///
/// This is deliberately not a serving framework. It is the smallest compile-time
/// boundary that connects Ferrule's resident scheduler spine:
///
/// `SchedulerAction -> ExecutionBatch -> TopKModelRunner -> ExecutionOutput`
///
/// A plain `TopKModelRunner` owns one resident backend session, so this executor
/// accepts chunked prefill and single-row decode. Real multi-session decode batches
/// should use a graph/backend executor that natively understands session ids and
/// paged KV handles; the scheduler/output contracts are already shared.
pub struct ResidentActionExecutor<R: TopKModelRunner> {
    runner: R,
    config: ResidentActionExecutorConfig,
}

impl<R: TopKModelRunner> ResidentActionExecutor<R> {
    pub fn new(runner: R) -> Self {
        Self::with_config(runner, ResidentActionExecutorConfig::default())
    }

    pub fn with_config(runner: R, config: ResidentActionExecutorConfig) -> Self {
        Self { runner, config }
    }

    pub fn config(&self) -> ResidentActionExecutorConfig {
        self.config
    }

    pub fn runner(&self) -> &R {
        &self.runner
    }

    pub fn runner_mut(&mut self) -> &mut R {
        &mut self.runner
    }

    pub fn into_runner(self) -> R {
        self.runner
    }

    pub fn position(&self) -> usize {
        self.runner.position()
    }

    pub fn execute_action(&mut self, action: &SchedulerAction) -> Result<Option<ExecutionOutput>> {
        let Some(batch) = action.execution_batch()? else {
            return Ok(None);
        };
        self.execute_batch(&batch).map(Some)
    }

    pub fn execute_batch(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput> {
        if self.config.top_k == 0 {
            return Err(Error::Internal("top_k must be greater than zero".into()));
        }
        match batch.segment {
            ExecutionSegment::Prefill => self.execute_prefill_batch(batch),
            ExecutionSegment::Decode => self.execute_decode_batch(batch),
            ExecutionSegment::Mixed => Err(Error::Internal(
                "TopK resident executor does not support mixed execution batches yet".into(),
            )),
        }
    }

    fn execute_prefill_batch(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput> {
        let rows = batch.rows();
        let first = rows[0];
        if first.position != self.runner.position() {
            return Err(Error::Internal(format!(
                "prefill position mismatch: runner at {}, batch starts at {}",
                self.runner.position(),
                first.position
            )));
        }

        let mut logits_row = None;
        for (index, row) in rows.iter().enumerate() {
            if row.session_id != first.session_id {
                return Err(Error::Internal(
                    "TopK resident executor cannot prefill multiple sessions in one batch".into(),
                ));
            }
            if row.kv_handle != first.kv_handle {
                return Err(Error::Internal(
                    "prefill batch rows must share the same KV handle".into(),
                ));
            }
            let expected_position = first.position + index;
            if row.position != expected_position {
                return Err(Error::Internal(format!(
                    "prefill batch positions must be contiguous: row {index} is {}, expected {expected_position}",
                    row.position
                )));
            }
            if row.require_logits {
                if logits_row.replace(index).is_some() {
                    return Err(Error::Internal(
                        "TopK resident executor can return logits for only one prefill row".into(),
                    ));
                }
            }
        }

        if let Some(index) = logits_row {
            if index + 1 != rows.len() {
                return Err(Error::Internal(
                    "TopK resident prefill can return logits only for the final prompt row".into(),
                ));
            }
        }

        let tokens: Vec<u32> = rows.iter().map(|row| row.token_id).collect();
        let topk = if logits_row.is_some() {
            Some(
                self.runner
                    .prefill_topk(&tokens, self.config.top_k, self.config.prefill_mode)?,
            )
        } else {
            self.runner
                .prefill_tokens(&tokens, self.config.prefill_mode)?;
            None
        };

        let output_rows = rows
            .iter()
            .enumerate()
            .map(|(index, row)| {
                let logits = if Some(index) == logits_row {
                    RowLogits::TopK(topk.clone().unwrap_or_default())
                } else {
                    RowLogits::None
                };
                ExecutionRowOutput::new(row.session_id, row.position, row.kv_handle, logits)
            })
            .collect();
        ExecutionOutput::new(output_rows)
    }

    fn execute_decode_batch(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput> {
        let rows = batch.rows();
        if rows.len() != 1 {
            return Err(Error::Internal(format!(
                "TopK resident executor can execute only one decode row, got {}",
                rows.len()
            )));
        }
        let row = rows[0];
        if row.position != self.runner.position() {
            return Err(Error::Internal(format!(
                "decode position mismatch: runner at {}, row at {}",
                self.runner.position(),
                row.position
            )));
        }

        let logits = if row.require_logits {
            RowLogits::TopK(self.runner.decode_topk(row.token_id, self.config.top_k)?)
        } else {
            self.runner.feed_token(row.token_id)?;
            RowLogits::None
        };

        ExecutionOutput::new(vec![ExecutionRowOutput::new(
            row.session_id,
            row.position,
            row.kv_handle,
            logits,
        )])
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use ferrule_common::Result;
    use ferrule_model::{ModelInfo, ModelRunner, TokenLogit};

    use crate::cache::{KvHandle, PagedSequenceKvCache};
    use crate::graph::runtime::{ExecutionRow, LogitsSelection};
    use crate::sampling::SamplingConfig;
    use crate::scheduling::{
        GenerateRequest, RequestId, ResidentScheduler, ResidentSchedulerConfig,
        SequenceFinishReason, SessionId,
    };

    use super::*;

    #[derive(Debug)]
    struct MockTopKRunner {
        position: usize,
        outputs: VecDeque<Vec<TokenLogit>>,
        fed: Vec<u32>,
        prefills: Vec<Vec<u32>>,
    }

    impl MockTopKRunner {
        fn new(outputs: Vec<Vec<TokenLogit>>) -> Self {
            Self {
                position: 0,
                outputs: outputs.into(),
                fed: Vec::new(),
                prefills: Vec::new(),
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
            self.prefills.clear();
            Ok(())
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
            Ok(())
        }

        fn prefill_topk(
            &mut self,
            token_ids: &[u32],
            _top_k: usize,
            _mode: PrefillMode,
        ) -> Result<Vec<TokenLogit>> {
            self.prefills.push(token_ids.to_vec());
            self.position += token_ids.len();
            Ok(self.next_output())
        }

        fn decode_topk(&mut self, token_id: u32, _top_k: usize) -> Result<Vec<TokenLogit>> {
            self.feed_token(token_id)?;
            Ok(self.next_output())
        }
    }

    fn top(token_id: u32) -> Vec<TokenLogit> {
        vec![TokenLogit {
            token_id,
            logit: 1.0,
        }]
    }

    fn request(id: u64, prompt_tokens: Vec<u32>, max_new_tokens: usize) -> GenerateRequest {
        GenerateRequest {
            id: RequestId(id),
            session_id: None,
            prompt_tokens,
            sampling: SamplingConfig::greedy(),
            max_new_tokens,
            stop: Vec::new(),
        }
    }

    #[test]
    fn executor_runs_prefill_batch_to_execution_output() {
        let runner = MockTopKRunner::new(vec![top(10)]);
        let mut executor = ResidentActionExecutor::new(runner);
        let batch = ExecutionBatch::from_tokens(
            ExecutionSegment::Prefill,
            SessionId(1),
            0,
            &[1, 2],
            Some(KvHandle(0)),
            LogitsSelection::Last,
        )
        .unwrap();

        let output = executor.execute_batch(&batch).unwrap();
        assert_eq!(executor.position(), 2);
        assert_eq!(output.rows().len(), 2);
        assert!(output.rows()[0].logits.is_none());
        assert_eq!(output.rows()[1].logits, RowLogits::TopK(top(10)));
    }

    #[test]
    fn executor_prefill_without_logits_uses_no_logits_runner_path() {
        let runner = MockTopKRunner::new(vec![top(99)]);
        let mut executor = ResidentActionExecutor::new(runner);
        let batch = ExecutionBatch::from_tokens(
            ExecutionSegment::Prefill,
            SessionId(1),
            0,
            &[1, 2],
            Some(KvHandle(0)),
            LogitsSelection::None,
        )
        .unwrap();

        let output = executor.execute_batch(&batch).unwrap();
        assert_eq!(executor.position(), 2);
        assert_eq!(output.rows().len(), 2);
        assert!(output.rows().iter().all(|row| row.logits.is_none()));
        assert!(executor.runner().prefills.is_empty());
        assert_eq!(executor.runner().fed, vec![1, 2]);
        assert_eq!(executor.runner().outputs.len(), 1);
    }

    #[test]
    fn executor_rejects_multi_row_decode_for_single_runner() {
        let runner = MockTopKRunner::new(vec![top(10)]);
        let mut executor = ResidentActionExecutor::new(runner);
        let batch = ExecutionBatch::new(
            ExecutionSegment::Decode,
            vec![
                ExecutionRow::new(10, 0, SessionId(1), None, true),
                ExecutionRow::new(20, 0, SessionId(2), None, true),
            ],
        )
        .unwrap();

        let err = executor.execute_batch(&batch).unwrap_err();
        assert!(format!("{err}").contains("only one decode row"));
    }

    #[test]
    fn scheduler_action_executor_output_chain_round_trips() {
        let mut scheduler = ResidentScheduler::new(ResidentSchedulerConfig {
            prefill_chunk_size: 2,
            max_active_sequences: 1,
            max_decode_batch: 1,
        });
        let mut kv = PagedSequenceKvCache::new(1, 1, 4, 1);
        let runner = MockTopKRunner::new(vec![top(10), top(11), top(12)]);
        let mut executor = ResidentActionExecutor::new(runner);
        scheduler.submit(request(7, vec![1, 2], 2));

        let prefill_action = scheduler
            .next_prefill_action(&mut kv)
            .unwrap()
            .expect("prefill action");
        let output = executor
            .execute_action(&prefill_action)
            .unwrap()
            .expect("prefill output");
        scheduler.commit_action(&prefill_action).unwrap();
        assert_eq!(
            scheduler.stage_greedy_decode_from_output(&output).unwrap(),
            1
        );

        let first_decode = scheduler
            .next_decode_action()
            .unwrap()
            .expect("first decode action");
        let output = executor
            .execute_action(&first_decode)
            .unwrap()
            .expect("decode output");
        scheduler.commit_action(&first_decode).unwrap();
        assert_eq!(
            scheduler.stage_greedy_decode_from_output(&output).unwrap(),
            1
        );

        let second_decode = scheduler
            .next_decode_action()
            .unwrap()
            .expect("second decode action");
        let output = executor
            .execute_action(&second_decode)
            .unwrap()
            .expect("decode output");
        scheduler.commit_action(&second_decode).unwrap();
        assert_eq!(
            scheduler.stage_greedy_decode_from_output(&output).unwrap(),
            0
        );

        let sequence = scheduler.active_sequence(SessionId(1)).unwrap();
        assert_eq!(sequence.tokens, vec![1, 2, 10, 11]);
        assert_eq!(sequence.generated, 2);
        assert_eq!(executor.position(), 4);

        scheduler
            .finish_sequence(SessionId(1), SequenceFinishReason::MaxTokens, &mut kv)
            .unwrap();
        assert_eq!(scheduler.finished_len(), 1);
        assert_eq!(kv.active_count(), 0);
    }
}
