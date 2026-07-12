use ferrule_common::execution::ExecutionOutput;
use ferrule_common::{Error, Result};
use ferrule_model::TopKModelRunner;

use crate::cache::SequenceKvCache;
use crate::generation::matched_stop;
use crate::scheduling::resident::greedy_candidate;
use crate::scheduling::{
    DecodeAction, GenerateRequest, ResidentScheduler, ResidentSchedulerConfig, ScheduledBatch,
    SchedulerAction, SequenceFinishReason, SequenceState, SessionId,
};

use super::{TopKCompatibilityExecutor, TopKCompatibilityExecutorConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentTopKDriverConfig {
    pub ctx_size: usize,
    pub stop_at_eos: bool,
    pub append_eos_to_session: bool,
    /// Safety valve for `run_until_blocked` so an unhealthy backend cannot spin forever.
    pub max_steps_per_run: usize,
}

impl Default for ResidentTopKDriverConfig {
    fn default() -> Self {
        Self {
            ctx_size: 4096,
            stop_at_eos: true,
            append_eos_to_session: true,
            max_steps_per_run: 16_384,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResidentTopKDriverStats {
    pub actions: usize,
    pub prefill_chunks: usize,
    pub prefill_tokens: usize,
    pub decode_steps: usize,
    pub emitted_tokens: usize,
    pub staged_tokens: usize,
    pub finished_sequences: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResidentTokenEvent {
    pub session_id: SessionId,
    pub request_id: Option<crate::scheduling::RequestId>,
    pub index: usize,
    pub token: u32,
    pub logit: Option<f32>,
    pub text: String,
    pub generated_text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResidentDriverStep {
    /// No waiting, active, or ready work remains.
    Idle,
    /// Work exists but no action could be produced, usually because KV admission is blocked.
    Blocked,
    /// One scheduler action was executed and committed.
    Executed {
        action_kind: ResidentActionKind,
        rows: usize,
        staged: usize,
        finished: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentActionKind {
    Prefill,
    Decode,
    Finish,
    Cancel,
}

/// Synchronous resident top-k driver over scheduler + KV + compatibility executor.
///
/// This is the first end-to-end serving-shaped loop in runtime. It remains concrete
/// and synchronous: no async server, no trait-object framework, and no concrete
/// model ownership. The driver exists to make the vLLM/SGLang spine executable:
/// request admission, chunked prefill, decode, stop policy, token streaming, and
/// KV/session finish lifecycle are all connected through typed runtime values.
pub struct ResidentTopKDriver<R, C>
where
    R: TopKModelRunner,
    C: SequenceKvCache,
{
    scheduler: ResidentScheduler,
    kv_cache: C,
    compatibility_executor: TopKCompatibilityExecutor<R>,
    config: ResidentTopKDriverConfig,
    stats: ResidentTopKDriverStats,
}

impl<R, C> ResidentTopKDriver<R, C>
where
    R: TopKModelRunner,
    C: SequenceKvCache,
{
    pub fn new(runner: R, kv_cache: C) -> Self {
        Self::with_parts(
            ResidentScheduler::default(),
            kv_cache,
            TopKCompatibilityExecutor::new(runner),
            ResidentTopKDriverConfig::default(),
        )
    }

    pub fn with_configs(
        runner: R,
        kv_cache: C,
        scheduler_config: ResidentSchedulerConfig,
        compatibility_executor_config: TopKCompatibilityExecutorConfig,
        driver_config: ResidentTopKDriverConfig,
    ) -> Self {
        Self::with_parts(
            ResidentScheduler::new(scheduler_config),
            kv_cache,
            TopKCompatibilityExecutor::with_config(runner, compatibility_executor_config),
            driver_config,
        )
    }

    fn with_parts(
        scheduler: ResidentScheduler,
        kv_cache: C,
        compatibility_executor: TopKCompatibilityExecutor<R>,
        config: ResidentTopKDriverConfig,
    ) -> Self {
        Self {
            scheduler,
            kv_cache,
            compatibility_executor,
            config,
            stats: ResidentTopKDriverStats::default(),
        }
    }

    pub fn scheduler(&self) -> &ResidentScheduler {
        &self.scheduler
    }

    pub fn kv_cache(&self) -> &C {
        &self.kv_cache
    }

    pub fn compatibility_executor(&self) -> &TopKCompatibilityExecutor<R> {
        &self.compatibility_executor
    }

    pub fn into_runner(self) -> Result<R> {
        if !self.scheduler.is_idle() {
            return Err(Error::Execution(
                "cannot extract resident runner while scheduler work is still active".into(),
            ));
        }
        self.compatibility_executor.into_runner()
    }

    pub fn stats(&self) -> &ResidentTopKDriverStats {
        &self.stats
    }

    /// Validate scheduler policy against the truthful capabilities of the legacy
    /// single-sequence compatibility executor before any queue entry is consumed.
    pub fn validate_configuration(&self) -> Result<()> {
        let capabilities = self.compatibility_executor.capabilities();
        let scheduler = self.scheduler.config();
        if scheduler.max_active_sequences > capabilities.max_sequences {
            return Err(Error::Execution(format!(
                "resident TopK driver config allows {} active sequences, but its compatibility executor supports {}",
                scheduler.max_active_sequences, capabilities.max_sequences
            )));
        }
        if scheduler.max_decode_batch > capabilities.max_sequences {
            return Err(Error::Execution(format!(
                "resident TopK driver config allows decode batch {}, but its compatibility executor supports {} sequence",
                scheduler.max_decode_batch, capabilities.max_sequences
            )));
        }
        let requested_top_k = self.compatibility_executor.default_top_k()?;
        if capabilities
            .max_top_k
            .is_some_and(|maximum| requested_top_k > maximum)
        {
            return Err(Error::Execution(format!(
                "resident TopK driver requests top-k {}, exceeding executor capability",
                requested_top_k.get()
            )));
        }
        Ok(())
    }

    pub fn submit(&mut self, request: GenerateRequest) {
        self.scheduler.submit(request);
    }

    /// Submit a request turn that appends to the current resident runner session.
    ///
    /// Use this for single-runner resident backends whose physical KV/session state
    /// lives inside the runner. It prevents the scheduler from planning the next
    /// prompt at position 0 after a previous turn advanced the runner.
    pub fn submit_at_current_position(&mut self, request: GenerateRequest) {
        let position_start = self.compatibility_executor.position();
        self.scheduler.submit_at_position(request, position_start);
    }

    pub fn drain_finished(&mut self) -> Vec<SequenceState> {
        self.scheduler.drain_finished()
    }

    pub fn drain_failed(&mut self) -> Vec<SequenceState> {
        self.scheduler.drain_failed()
    }

    pub fn step<F>(&mut self, on_token: &mut F) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        self.validate_configuration()?;
        let Some(mut action) = self.scheduler.next_action(&mut self.kv_cache)? else {
            return Ok(if self.scheduler.is_idle() {
                ResidentDriverStep::Idle
            } else {
                ResidentDriverStep::Blocked
            });
        };

        let action_kind = action_kind(&action);
        let rows = action_rows(&action);
        let scheduled = match ScheduledBatch::from_action(
            &mut action,
            self.compatibility_executor.default_top_k()?,
        ) {
            Ok(scheduled) => scheduled,
            Err(error) => return Err(self.abort_action(&action, error, false, "batch lowering")),
        };
        let output = match scheduled.as_ref() {
            Some(batch) => match self.compatibility_executor.execute_batch(batch.execution()) {
                Ok(output) => Some(output),
                Err(execution_error) => {
                    let poisoned = self.compatibility_executor.is_poisoned();
                    return Err(self.abort_action(
                        &action,
                        execution_error,
                        poisoned,
                        "model execution",
                    ));
                }
            },
            None => None,
        };

        if let (Some(batch), Some(output)) = (scheduled.as_ref(), output.as_ref()) {
            if let Err(contract_error) = batch.validate_output(output) {
                return Err(self.abort_action(
                    &action,
                    contract_error,
                    true,
                    "model output contract",
                ));
            }
        } else if scheduled.is_some() != output.is_some() {
            let error =
                Error::Internal("scheduled execution and model output presence diverged".into());
            return Err(self.abort_action(&action, error, true, "model output presence"));
        }

        if let Err(commit_error) = self.scheduler.commit_action(&action) {
            return Err(self.abort_action(&action, commit_error, true, "runtime commit"));
        }
        self.stats.actions += 1;
        match &action {
            SchedulerAction::PrefillChunk(prefill) => {
                self.stats.prefill_chunks += 1;
                self.stats.prefill_tokens += prefill.token_range.len();
            }
            SchedulerAction::DecodeBatch(actions) => {
                self.stats.decode_steps += actions.len();
                if let Err(error) = self.emit_committed_decode_tokens(actions, on_token) {
                    return Err(self.abort_action(&action, error, true, "token emission"));
                }
            }
            SchedulerAction::Finish { .. } | SchedulerAction::Cancel { .. } => {}
        }

        let action_finish = match self.finish_after_decode_action(&action) {
            Ok(outcome) => outcome,
            Err(error) => {
                return Err(self.abort_action(&action, error, true, "sequence finish"));
            }
        };
        let mut finished = action_finish.finished;
        let staged = match (scheduled.as_ref(), output.as_ref()) {
            (Some(batch), Some(output)) => {
                let outcome =
                    match self.apply_execution_output(batch, output, &action_finish.session_ids) {
                        Ok(outcome) => outcome,
                        Err(error) => {
                            return Err(self.abort_action(
                                &action,
                                error,
                                true,
                                "output application",
                            ));
                        }
                    };
                finished += outcome.finished;
                outcome.staged
            }
            (None, None) => 0,
            _ => unreachable!("scheduled/output presence was validated before commit"),
        };

        Ok(ResidentDriverStep::Executed {
            action_kind,
            rows,
            staged,
            finished,
        })
    }

    fn abort_action(
        &mut self,
        action: &SchedulerAction,
        error: Error,
        poison_executor: bool,
        stage: &'static str,
    ) -> Error {
        if poison_executor && !self.compatibility_executor.is_poisoned() {
            self.compatibility_executor
                .poison_after_output_contract(&error);
        }
        match self.scheduler.fail_action(action, &mut self.kv_cache) {
            Ok(_) => error,
            Err(cleanup_error) => Error::Internal(format!(
                "{stage} failed ({error}); error-state cleanup also failed ({cleanup_error})"
            )),
        }
    }

    pub fn run_until_blocked<F>(&mut self, mut on_token: F) -> Result<ResidentTopKDriverStats>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        for _ in 0..self.config.max_steps_per_run {
            match self.step(&mut on_token)? {
                ResidentDriverStep::Executed { .. } => {}
                ResidentDriverStep::Idle | ResidentDriverStep::Blocked => {
                    return Ok(self.stats.clone())
                }
            }
        }
        Err(Error::Internal(format!(
            "resident top-k driver exceeded max_steps_per_run={} without becoming idle or blocked",
            self.config.max_steps_per_run
        )))
    }

    fn emit_committed_decode_tokens<F>(
        &mut self,
        actions: &[DecodeAction],
        on_token: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()>,
    {
        for action in actions {
            let text = self
                .compatibility_executor
                .runner()
                .decode(&[action.token_id])?;
            let sequence = self
                .scheduler
                .active_sequence_mut(action.session_id)
                .ok_or_else(|| {
                    Error::Internal(format!(
                        "cannot emit token for inactive session {:?}",
                        action.session_id
                    ))
                })?;
            sequence.append_generated_text(&text);
            let index = sequence.generated.saturating_sub(1);
            let event = ResidentTokenEvent {
                session_id: sequence.session_id,
                request_id: sequence.request_id,
                index,
                token: action.token_id,
                logit: action.logit,
                text,
                generated_text: sequence.generated_text.clone(),
            };
            self.stats.emitted_tokens += 1;
            on_token(&event)?;
        }
        Ok(())
    }

    fn finish_after_decode_action(
        &mut self,
        action: &SchedulerAction,
    ) -> Result<ActionFinishOutcome> {
        let SchedulerAction::DecodeBatch(actions) = action else {
            return Ok(ActionFinishOutcome::default());
        };

        let mut outcome = ActionFinishOutcome::default();
        for action in actions {
            let Some(sequence) = self.scheduler.active_sequence(action.session_id) else {
                continue;
            };
            let reason = if sequence.generated >= sequence.max_new_tokens {
                Some(SequenceFinishReason::MaxTokens)
            } else if matched_stop(&sequence.generated_text, &sequence.stop).is_some() {
                Some(SequenceFinishReason::StopString)
            } else {
                None
            };
            if let Some(reason) = reason {
                self.scheduler
                    .finish_sequence(action.session_id, reason, &mut self.kv_cache)?;
                self.stats.finished_sequences += 1;
                outcome.finished += 1;
                outcome.session_ids.push(action.session_id);
            }
        }
        Ok(outcome)
    }

    fn apply_execution_output(
        &mut self,
        scheduled: &ScheduledBatch,
        output: &ExecutionOutput,
        action_finished_sessions: &[SessionId],
    ) -> Result<OutputOutcome> {
        let mut outcome = OutputOutcome::default();
        for row in &output.logits {
            let correlation = scheduled
                .sequence_for_input_row(row.input_row)
                .copied()
                .ok_or_else(|| {
                    Error::Execution(format!(
                        "output input row {} has no scheduled sequence",
                        row.input_row
                    ))
                })?;
            let execution_sequence = scheduled
                .execution()
                .sequences()
                .iter()
                .find(|sequence| sequence.query.contains(&row.input_row))
                .ok_or_else(|| {
                    Error::Execution(format!(
                        "output input row {} has no execution sequence span",
                        row.input_row
                    ))
                })?;
            let session_id = correlation.session_id;
            let Some(sequence) = self.scheduler.active_sequence(session_id) else {
                if action_finished_sessions.contains(&session_id) {
                    // The just-committed token ended the sequence (for example via
                    // a stop string), so its already-computed next-token logits are
                    // intentionally discarded after successful correlation.
                    continue;
                }
                return Err(Error::Execution(format!(
                    "output for input row {} references inactive session {:?}",
                    row.input_row, session_id
                )));
            };
            if sequence.request_id != correlation.request_id {
                return Err(Error::Execution(format!(
                    "output correlation request mismatch for session {:?}: active {:?}, scheduled {:?}",
                    session_id, sequence.request_id, correlation.request_id
                )));
            }
            if sequence.kv_handle != correlation.kv_handle {
                return Err(Error::Execution(format!(
                    "output correlation KV mismatch for session {:?}: active {:?}, scheduled {:?}",
                    session_id, sequence.kv_handle, correlation.kv_handle
                )));
            }
            if sequence.position != execution_sequence.sequence_len as usize {
                return Err(Error::Execution(format!(
                    "output correlation position mismatch for session {:?}: active {}, executed {}",
                    session_id, sequence.position, execution_sequence.sequence_len
                )));
            }
            if sequence.generated >= sequence.max_new_tokens {
                self.scheduler.finish_sequence(
                    session_id,
                    SequenceFinishReason::MaxTokens,
                    &mut self.kv_cache,
                )?;
                self.stats.finished_sequences += 1;
                outcome.finished += 1;
                continue;
            }
            if sequence.position >= self.config.ctx_size {
                self.scheduler.finish_sequence(
                    session_id,
                    SequenceFinishReason::Context,
                    &mut self.kv_cache,
                )?;
                self.stats.finished_sequences += 1;
                outcome.finished += 1;
                continue;
            }

            let Some(candidate) = greedy_candidate(&row.logits) else {
                self.scheduler.finish_sequence(
                    session_id,
                    SequenceFinishReason::NoCandidate,
                    &mut self.kv_cache,
                )?;
                self.stats.finished_sequences += 1;
                outcome.finished += 1;
                continue;
            };

            if self.config.stop_at_eos
                && self.compatibility_executor.runner().eos_token_id() == Some(candidate.token_id)
            {
                if self.config.append_eos_to_session {
                    if let Err(execution_error) =
                        self.compatibility_executor.feed_token(candidate.token_id)
                    {
                        if let Err(cleanup_error) =
                            self.scheduler.fail_sequence(session_id, &mut self.kv_cache)
                        {
                            return Err(Error::Internal(format!(
                                "EOS state update failed ({execution_error}); error-state cleanup also failed ({cleanup_error})"
                            )));
                        }
                        return Err(execution_error);
                    }
                    if let Some(sequence) = self.scheduler.active_sequence_mut(session_id) {
                        sequence.advance_position(1);
                    }
                }
                self.scheduler.finish_sequence(
                    session_id,
                    SequenceFinishReason::Eos,
                    &mut self.kv_cache,
                )?;
                self.stats.finished_sequences += 1;
                outcome.finished += 1;
                continue;
            }

            self.scheduler.stage_decode_candidate(
                session_id,
                candidate.token_id,
                Some(candidate.logit),
            )?;
            self.stats.staged_tokens += 1;
            outcome.staged += 1;
        }
        Ok(outcome)
    }
}

#[derive(Default)]
struct ActionFinishOutcome {
    finished: usize,
    session_ids: Vec<SessionId>,
}

#[derive(Default)]
struct OutputOutcome {
    staged: usize,
    finished: usize,
}

fn action_kind(action: &SchedulerAction) -> ResidentActionKind {
    match action {
        SchedulerAction::PrefillChunk(_) => ResidentActionKind::Prefill,
        SchedulerAction::DecodeBatch(_) => ResidentActionKind::Decode,
        SchedulerAction::Finish { .. } => ResidentActionKind::Finish,
        SchedulerAction::Cancel { .. } => ResidentActionKind::Cancel,
    }
}

fn action_rows(action: &SchedulerAction) -> usize {
    match action {
        SchedulerAction::PrefillChunk(prefill) => prefill.token_range.len(),
        SchedulerAction::DecodeBatch(actions) => actions.len(),
        SchedulerAction::Finish { .. } | SchedulerAction::Cancel { .. } => 0,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use ferrule_common::{Error, Result};
    use ferrule_model::{ModelInfo, ModelRunner, PrefillMode, TokenLogit};

    use crate::cache::PagedSequenceKvCache;
    use crate::sampling::SamplingConfig;
    use crate::scheduling::{RequestId, SequenceStatus};

    use super::*;

    #[derive(Debug)]
    struct MockTopKRunner {
        position: usize,
        eos: Option<u32>,
        outputs: VecDeque<Vec<TokenLogit>>,
        fed: Vec<u32>,
        prefills: Vec<Vec<u32>>,
        fail_next_mutation: bool,
        mutation_calls: usize,
    }

    impl MockTopKRunner {
        fn new(outputs: Vec<Vec<TokenLogit>>) -> Self {
            Self {
                position: 0,
                eos: None,
                outputs: outputs.into(),
                fed: Vec::new(),
                prefills: Vec::new(),
                fail_next_mutation: false,
                mutation_calls: 0,
            }
        }

        fn failing_next_mutation(mut self) -> Self {
            self.fail_next_mutation = true;
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

        fn with_eos(mut self, eos: u32) -> Self {
            self.eos = Some(eos);
            self
        }

        fn next_output(&mut self) -> Vec<TokenLogit> {
            self.outputs.pop_front().unwrap_or_default()
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
                vocab_size: 256,
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
            self.eos
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

        fn prefill_topk(
            &mut self,
            token_ids: &[u32],
            _top_k: usize,
            _mode: PrefillMode,
        ) -> Result<Vec<TokenLogit>> {
            self.prefills.push(token_ids.to_vec());
            self.position += token_ids.len();
            let output = self.next_output();
            self.complete_mutation(output)
        }

        fn decode_topk(&mut self, token_id: u32, _top_k: usize) -> Result<Vec<TokenLogit>> {
            self.feed_token(token_id)?;
            Ok(self.next_output())
        }
    }

    fn top(token_id: u32) -> Vec<TokenLogit> {
        vec![TokenLogit {
            token_id,
            logit: token_id as f32,
        }]
    }

    fn request(
        id: u64,
        prompt: &[u32],
        max_new_tokens: usize,
        stop: Vec<String>,
    ) -> GenerateRequest {
        GenerateRequest {
            id: RequestId(id),
            session_id: None,
            prompt_tokens: prompt.to_vec(),
            sampling: SamplingConfig::greedy(),
            max_new_tokens,
            stop,
        }
    }

    fn driver_from_runner(
        runner: MockTopKRunner,
    ) -> ResidentTopKDriver<MockTopKRunner, PagedSequenceKvCache> {
        ResidentTopKDriver::with_configs(
            runner,
            PagedSequenceKvCache::new(1, 1, 8, 1),
            ResidentSchedulerConfig {
                prefill_chunk_size: 2,
                max_active_sequences: 1,
                max_decode_batch: 1,
            },
            TopKCompatibilityExecutorConfig {
                top_k: 1,
                prefill_mode: PrefillMode::Interactive,
            },
            ResidentTopKDriverConfig {
                ctx_size: 16,
                stop_at_eos: true,
                append_eos_to_session: true,
                max_steps_per_run: 64,
            },
        )
    }

    fn driver_with_outputs(
        outputs: Vec<Vec<TokenLogit>>,
    ) -> ResidentTopKDriver<MockTopKRunner, PagedSequenceKvCache> {
        driver_from_runner(MockTopKRunner::new(outputs))
    }

    #[test]
    fn driver_runs_request_to_max_tokens_and_frees_kv() {
        let mut driver = driver_with_outputs(vec![top(b'a' as u32), top(b'b' as u32)]);
        driver.submit(request(1, &[1, 2], 2, Vec::new()));
        let mut events = Vec::new();
        let stats = driver
            .run_until_blocked(|event| {
                events.push(event.clone());
                Ok(())
            })
            .unwrap();

        assert_eq!(
            events.iter().map(|event| event.token).collect::<Vec<_>>(),
            vec![b'a' as u32, b'b' as u32]
        );
        assert_eq!(events[0].text, "a");
        assert_eq!(events[1].generated_text, "ab");
        assert_eq!(stats.prefill_chunks, 1);
        assert_eq!(stats.prefill_tokens, 2);
        assert_eq!(stats.decode_steps, 2);
        assert_eq!(stats.emitted_tokens, 2);
        assert_eq!(stats.finished_sequences, 1);
        assert_eq!(driver.kv_cache().active_count(), 0);

        let finished = driver.drain_finished();
        assert_eq!(finished.len(), 1);
        assert_eq!(
            finished[0].finish_reason,
            Some(SequenceFinishReason::MaxTokens)
        );
        assert_eq!(finished[0].generated_text, "ab");
        assert_eq!(finished[0].status, SequenceStatus::Finished);
    }

    #[test]
    fn driver_stops_on_stop_string_after_committed_token() {
        let mut driver =
            driver_with_outputs(vec![top(b'a' as u32), top(b'b' as u32), top(b'c' as u32)]);
        driver.submit(request(2, &[9], 8, vec!["ab".into()]));
        let mut text = String::new();
        driver
            .run_until_blocked(|event| {
                text.push_str(&event.text);
                Ok(())
            })
            .unwrap();

        let finished = driver.drain_finished();
        assert_eq!(text, "ab");
        assert_eq!(
            finished[0].finish_reason,
            Some(SequenceFinishReason::StopString)
        );
        assert_eq!(finished[0].generated_text, "ab");
    }

    #[test]
    fn driver_finishes_on_eos_without_emitting_token() {
        let runner = MockTopKRunner::new(vec![top(2)]).with_eos(2);
        let mut driver = ResidentTopKDriver::with_configs(
            runner,
            PagedSequenceKvCache::new(1, 1, 4, 1),
            ResidentSchedulerConfig::default(),
            TopKCompatibilityExecutorConfig::default(),
            ResidentTopKDriverConfig::default(),
        );
        driver.submit(request(3, &[1], 4, Vec::new()));
        let mut events = Vec::new();
        driver
            .run_until_blocked(|event| {
                events.push(event.clone());
                Ok(())
            })
            .unwrap();

        assert!(events.is_empty());
        assert_eq!(driver.compatibility_executor().runner().position(), 2);
        let finished = driver.drain_finished();
        assert_eq!(finished[0].finish_reason, Some(SequenceFinishReason::Eos));
        assert_eq!(finished[0].tokens, vec![1]);
    }

    #[test]
    fn final_decode_skips_next_logits() {
        let mut driver = driver_with_outputs(vec![top(b'a' as u32)]);
        driver.submit(request(4, &[1], 1, Vec::new()));
        driver.run_until_blocked(|_| Ok(())).unwrap();

        assert_eq!(driver.compatibility_executor().runner().outputs.len(), 0);
        assert_eq!(
            driver.compatibility_executor().runner().fed,
            vec![b'a' as u32]
        );
        let finished = driver.drain_finished();
        assert_eq!(
            finished[0].finish_reason,
            Some(SequenceFinishReason::MaxTokens)
        );
    }

    #[test]
    fn max_new_zero_finishes_after_prefill() {
        let mut driver = driver_with_outputs(vec![top(b'a' as u32)]);
        driver.submit(request(5, &[1], 0, Vec::new()));
        driver.run_until_blocked(|_| Ok(())).unwrap();

        assert!(driver.compatibility_executor().runner().fed.is_empty());
        let finished = driver.drain_finished();
        assert_eq!(
            finished[0].finish_reason,
            Some(SequenceFinishReason::MaxTokens)
        );
        assert_eq!(finished[0].tokens, vec![1]);
    }

    #[test]
    fn submit_at_current_position_continues_resident_runner_session() {
        let mut driver = driver_with_outputs(vec![top(b'a' as u32), top(b'b' as u32)]);
        driver.submit_at_current_position(request(7, &[1], 1, Vec::new()));
        driver.run_until_blocked(|_| Ok(())).unwrap();
        let first = driver.drain_finished();
        assert_eq!(first[0].position, 2);
        assert_eq!(driver.compatibility_executor().position(), 2);

        driver.submit_at_current_position(request(8, &[2], 1, Vec::new()));
        driver.run_until_blocked(|_| Ok(())).unwrap();
        let second = driver.drain_finished();
        assert_eq!(second[0].prompt_tokens_for_range(0..1).unwrap(), &[2]);
        assert_eq!(second[0].position, 4);
        assert_eq!(driver.compatibility_executor().position(), 4);
    }

    #[test]
    fn into_runner_allows_clean_driver_rebuild_after_warmup() {
        let mut driver = driver_with_outputs(vec![top(b'w' as u32), top(b'm' as u32)]);
        driver.submit_at_current_position(request(9, &[1], 1, Vec::new()));
        driver.run_until_blocked(|_| Ok(())).unwrap();
        let warmup = driver.drain_finished();
        assert_eq!(warmup[0].position, 2);
        assert_eq!(driver.compatibility_executor().position(), 2);

        let mut runner = driver.into_runner().unwrap();
        runner.reset_session().unwrap();
        let mut driver = driver_from_runner(runner);
        driver.submit_at_current_position(request(10, &[2], 1, Vec::new()));
        driver.run_until_blocked(|_| Ok(())).unwrap();
        let measured = driver.drain_finished();

        assert_eq!(measured[0].prompt_tokens_for_range(0..1).unwrap(), &[2]);
        assert_eq!(measured[0].position, 2);
        assert_eq!(driver.compatibility_executor().position(), 2);
    }

    #[test]
    fn driver_moves_partially_executed_sequence_to_error_state() {
        let runner = MockTopKRunner::new(vec![top(b'a' as u32)]).failing_next_mutation();
        let mut driver = driver_from_runner(runner);
        driver.submit(request(11, &[1, 2], 1, Vec::new()));

        let error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(format!("{error}").contains("simulated failure"));
        assert!(driver.compatibility_executor().is_poisoned());
        assert_eq!(driver.compatibility_executor().position(), 2);
        assert_eq!(driver.compatibility_executor().runner().mutation_calls, 1);
        assert_eq!(driver.scheduler().active_len(), 0);
        assert_eq!(driver.scheduler().failed_len(), 1);
        assert_eq!(driver.kv_cache().active_count(), 0);

        let failed = driver.drain_failed();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].status, SequenceStatus::Error);
        assert_eq!(failed[0].position, 0, "runtime metadata was not committed");
        assert_eq!(failed[0].kv_handle, None);

        driver.submit_at_current_position(request(12, &[3], 1, Vec::new()));
        let error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(format!("{error}").contains("resident executor is poisoned"));
        assert_eq!(driver.compatibility_executor().runner().mutation_calls, 1);
        let failed = driver.drain_failed();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].status, SequenceStatus::Error);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn lowering_failure_moves_dequeued_sequence_to_failed_state() {
        let mut runner = MockTopKRunner::new(Vec::new());
        runner.position = u32::MAX as usize + 1;
        let mut driver = driver_from_runner(runner);
        driver.submit_at_current_position(request(13, &[1], 1, Vec::new()));

        let error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(format!("{error}").contains("neutral u32 ABI"));
        assert!(!driver.compatibility_executor().is_poisoned());
        assert_eq!(driver.scheduler().active_len(), 0);
        assert_eq!(driver.scheduler().failed_len(), 1);
        assert_eq!(driver.kv_cache().active_count(), 0);
        assert_eq!(driver.compatibility_executor().runner().mutation_calls, 0);
    }

    #[test]
    fn callback_failure_poison_fails_committed_sequence_and_blocks_runner_extraction() {
        let mut driver = driver_with_outputs(vec![top(b'a' as u32), top(b'b' as u32)]);
        driver.submit(request(14, &[1], 3, Vec::new()));

        let error = driver
            .run_until_blocked(|_| Err(Error::Internal("simulated callback failure".into())))
            .unwrap_err();
        assert!(format!("{error}").contains("callback failure"));
        assert!(driver.compatibility_executor().is_poisoned());
        assert_eq!(driver.scheduler().active_len(), 0);
        assert_eq!(driver.scheduler().failed_len(), 1);
        assert_eq!(driver.kv_cache().active_count(), 0);
        assert!(driver.into_runner().is_err());
    }

    #[test]
    fn driver_rejects_unsupported_scheduler_width_before_consuming_work() {
        let mut driver = ResidentTopKDriver::with_configs(
            MockTopKRunner::new(Vec::new()),
            PagedSequenceKvCache::new(1, 1, 4, 2),
            ResidentSchedulerConfig {
                prefill_chunk_size: 2,
                max_active_sequences: 2,
                max_decode_batch: 2,
            },
            TopKCompatibilityExecutorConfig::default(),
            ResidentTopKDriverConfig::default(),
        );
        driver.submit(request(13, &[1, 2], 1, Vec::new()));

        let error = driver.step(&mut |_| Ok(())).unwrap_err();
        assert!(format!("{error}").contains("allows 2 active sequences"));
        assert_eq!(driver.scheduler().waiting_len(), 1);
        assert_eq!(driver.scheduler().active_len(), 0);
        assert_eq!(driver.kv_cache().active_count(), 0);
        assert_eq!(driver.compatibility_executor().runner().mutation_calls, 0);
    }

    #[test]
    fn driver_blocks_when_kv_cannot_admit_waiting_request() {
        let mut driver = ResidentTopKDriver::with_configs(
            MockTopKRunner::new(Vec::new()),
            PagedSequenceKvCache::new(1, 1, 1, 0),
            ResidentSchedulerConfig::default(),
            TopKCompatibilityExecutorConfig::default(),
            ResidentTopKDriverConfig::default(),
        );
        driver.submit(request(6, &[1], 1, Vec::new()));
        let step = driver.step(&mut |_| Ok(())).unwrap();
        assert_eq!(step, ResidentDriverStep::Blocked);
        assert_eq!(driver.scheduler().waiting_len(), 1);
    }
}
