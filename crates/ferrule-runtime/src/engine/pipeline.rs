//! Serving policy over the pipeline's sole logical and physical KV lifecycle.
//!
//! Scheduler slots are request bookkeeping, never pipeline page slots. Execution
//! is serial and cancellation is cooperative at committed chunk/token boundaries;
//! this adapter does not invent async continuations or a second transaction owner.

use std::collections::{HashMap, HashSet, VecDeque};

use ferrule_common::CompletionHub;
use ferrule_common::execution::{ForwardPhase, LogitsOutput};
use ferrule_model::{ModelInfo, TokenizerHandle};

use crate::parallel::pipeline::PipelineParallelExecutor;
use crate::scheduling::resident::greedy_candidate;
use crate::scheduling::{
    FixedSequenceSlotPool, GenerateRequest, LogitsSelection, RequestId, RequestTerminal,
    ResidentScheduler, ResidentSchedulerConfig, SchedulerAction, SequenceFinishReason,
    SequenceState, SessionId,
};
use crate::{Error, Result};

use super::{
    InferenceCancelProgress, InferenceCompletionReactor, InferenceEngine,
    InferenceShutdownProgress, ResidentActionKind, ResidentDriverStep, ResidentEngineObservability,
    ResidentKvCacheObservability, ResidentKvPagePlan, ResidentTokenEvent, ResidentTopKDriverStats,
    SessionInferenceEngine,
};

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidRequest {
        message: message.into(),
    }
}

/// Owner-local CPU/CUDA serving adapter. The pipeline alone owns pages, COW,
/// distributed decisions, physical fences and retirement acknowledgements.
pub struct PipelineInferenceEngine {
    pipeline: PipelineParallelExecutor,
    tokenizer: TokenizerHandle,
    info: ModelInfo,
    scheduler: ResidentScheduler,
    slots: FixedSequenceSlotPool,
    requests: HashMap<RequestId, SessionId>,
    cleanup_receipts: HashMap<RequestId, (super::RequestCleanupOwner, bool)>,
    retained: HashSet<SessionId>,
    text: HashMap<SessionId, StopText>,
    outbox: VecDeque<ResidentTokenEvent>,
    pending_terminals: HashSet<RequestId>,
    rejected: Vec<SequenceState>,
    stop_at_eos: bool,
    stats: ResidentTopKDriverStats,
    kv_plan: ResidentKvPagePlan,
    completion: CompletionHub,
    closed: bool,
    faulted: bool,
}

impl PipelineInferenceEngine {
    pub fn new(
        pipeline: PipelineParallelExecutor,
        tokenizer: TokenizerHandle,
        info: ModelInfo,
        scheduler: ResidentSchedulerConfig,
        stop_at_eos: bool,
        kv_plan: ResidentKvPagePlan,
    ) -> Result<Self> {
        let config = pipeline.config();
        if pipeline.is_quarantined()
            || pipeline.outstanding() != 0
            || pipeline.page_manager().active_sequences() != 0
        {
            return Err(invalid("serving requires a fresh, quiescent pipeline"));
        }
        let full_pages = config
            .max_positions
            .div_ceil(config.page_size)
            .checked_mul(config.session_capacity)
            .ok_or_else(|| invalid("pipeline serving page capacity overflow"))?;
        if scheduler.max_active_sequences != config.session_capacity
            || scheduler.max_decode_batch != 1
            || scheduler.allow_mixed_batches
            || scheduler.decode_cohort_target != 1
            || scheduler.decode_cohort_max_deferrals != 0
            || scheduler.prefix_cache_capacity_pages != 0
            || scheduler.prefill_chunk_size == 0
            || scheduler.max_batch_tokens == 0
            || scheduler.max_batch_tokens > config.max_batch_tokens
            || scheduler.prefill_chunk_size > config.max_batch_tokens
        {
            return Err(invalid(
                "pipeline serving requires serial, non-mixed scheduling with matching bounded capacities and no prefix cache",
            ));
        }
        if config.max_pages < full_pages
            || kv_plan.full_capacity_pages != config.max_pages
            || kv_plan.full_capacity_pages != full_pages
        {
            return Err(invalid(
                "pipeline serving KV budget must cover every resident session's full context; overcommit is unsupported",
            ));
        }
        let first = pipeline
            .stage_descriptions()
            .first()
            .ok_or_else(|| invalid("pipeline has no stages"))?;
        if kv_plan.configured_pages != first.physical_pages {
            return Err(invalid("pipeline physical capacity differs from KV budget"));
        }
        let physical_bytes =
            pipeline
                .stage_descriptions()
                .iter()
                .try_fold(0u64, |sum, stage| {
                    sum.checked_add(stage.physical_bytes()?)
                        .ok_or_else(|| invalid("pipeline physical byte total overflow"))
                })?;
        if kv_plan
            .configured_bytes
            .is_some_and(|bytes| bytes != physical_bytes)
        {
            return Err(invalid("pipeline physical bytes differ from KV budget"));
        }
        if info.vocab_size != first.vocabulary || info.num_layers != first.plan.total_layers() {
            return Err(invalid(
                "pipeline serving model metadata differs from its stages",
            ));
        }
        Ok(Self {
            pipeline,
            tokenizer,
            info,
            scheduler: ResidentScheduler::new(scheduler),
            slots: FixedSequenceSlotPool::new(config.session_capacity),
            requests: HashMap::new(),
            cleanup_receipts: HashMap::new(),
            retained: HashSet::new(),
            text: HashMap::new(),
            outbox: VecDeque::new(),
            pending_terminals: HashSet::new(),
            rejected: Vec::new(),
            stop_at_eos,
            stats: Default::default(),
            kv_plan,
            completion: CompletionHub::new(),
            closed: false,
            faulted: false,
        })
    }

    /// Read-only access cannot allocate, publish or retire pipeline pages.
    pub fn pipeline(&self) -> &PipelineParallelExecutor {
        &self.pipeline
    }

    fn available(&self) -> Result<()> {
        if self.closed || self.faulted || self.pipeline.is_quarantined() {
            return Err(Error::EngineUnavailable);
        }
        Ok(())
    }

    fn position(&self, session: SessionId) -> Option<usize> {
        let slot = self.pipeline.session_slot(session)?;
        self.pipeline
            .page_manager()
            .block_table(slot)
            .map(|table| table.committed_tokens())
    }

    fn idle_session(&self, session: SessionId) -> Result<()> {
        self.available()?;
        if self.requests.values().any(|&active| active == session) {
            return Err(invalid("session still owns an active request"));
        }
        Ok(())
    }

    /// Fork only the complete committed prefix of an idle retained session.
    /// No HTTP multi-choice or implicit prefix-cache behavior is introduced.
    pub fn fork_session_exact(
        &mut self,
        source: SessionId,
        target: SessionId,
        expected_position: usize,
    ) -> Result<()> {
        self.idle_session(source)?;
        if !self.retained.contains(&source) || self.position(source) != Some(expected_position) {
            return Err(invalid("fork source is not an exact idle retained prefix"));
        }
        if source == target
            || self.retained.contains(&target)
            || self.pipeline.session_slot(target).is_some()
        {
            return Err(invalid("fork target already exists"));
        }
        self.check_session_capacity()?;
        self.pipeline.fork_session(source, target)?;
        self.retained.insert(target);
        Ok(())
    }

    pub fn release_session(&mut self, session: SessionId) -> Result<()> {
        self.idle_session(session)?;
        if self.pipeline.session_slot(session).is_some() {
            self.pipeline.release_session(session)?;
        }
        self.retained.remove(&session);
        Ok(())
    }

    fn check_session_capacity(&self) -> Result<()> {
        if self.pipeline.page_manager().active_sequences()
            >= self.pipeline.config().session_capacity
        {
            return Err(Error::RequestCapacity {
                limit: self.pipeline.config().session_capacity,
            });
        }
        Ok(())
    }

    fn flush(&mut self, on_token: &mut dyn FnMut(&ResidentTokenEvent) -> Result<()>) -> Result<()> {
        while let Some(event) = self.outbox.front() {
            on_token(event)?;
            self.outbox.pop_front();
            self.stats.emitted_tokens += 1;
        }
        Ok(())
    }

    fn finish(&mut self, session: SessionId, reason: SequenceFinishReason) -> Result<()> {
        if let Some(output) = self.text.get_mut(&session) {
            output.finish(&mut self.outbox);
        }
        // A terminal request must not make a physical session reusable before
        // every pipeline owner has acknowledged its release.
        if !self.retained.contains(&session) {
            self.pipeline.release_session(session)?;
        }
        let request = self
            .scheduler
            .active_sequence(session)
            .and_then(|s| s.request_id);
        self.scheduler
            .finish_sequence(session, reason, &mut self.slots)?;
        if let Some(request) = request {
            self.requests.remove(&request);
            self.pending_terminals.insert(request);
        }
        self.text.remove(&session);
        self.stats.finished_sequences += 1;
        Ok(())
    }

    fn reap_cleanup_receipts(&mut self) {
        // `requests` is removed only after the pipeline's physical release
        // succeeds (or ownership remains with an explicitly retained session).
        // Never infer physical return from a terminal, idle, or fault flag.
        let released = self
            .cleanup_receipts
            .iter()
            .filter_map(|(request, (_, consumed))| {
                (*consumed
                    && !self.requests.contains_key(request)
                    && !self.pending_terminals.contains(request)
                    && !self
                        .outbox
                        .iter()
                        .any(|event| event.request_id == Some(*request)))
                .then_some(*request)
            })
            .collect::<Vec<_>>();
        for request in released {
            let (owner, _) = self
                .cleanup_receipts
                .remove(&request)
                .expect("cleanup owner");
            owner.release();
        }
    }

    fn consume_cleanup_terminal(&mut self, request: RequestId) {
        if let Some((_, consumed)) = self.cleanup_receipts.get_mut(&request) {
            *consumed = true;
        }
    }

    fn consume_terminals(&mut self, sequences: &[SequenceState]) {
        for sequence in sequences {
            if let Some(request) = sequence.request_id {
                self.pending_terminals.remove(&request);
                self.consume_cleanup_terminal(request);
            }
        }
    }

    fn next_candidate(&mut self, session: SessionId, row: &[f32]) -> Result<usize> {
        if row.len() != self.info.vocab_size || row.iter().any(|value| !value.is_finite()) {
            return Err(Error::Invariant {
                message: "pipeline returned invalid vocabulary logits".into(),
            });
        }
        let sequence = self
            .scheduler
            .active_sequence(session)
            .ok_or_else(|| invalid("missing scheduled pipeline session"))?;
        let reason = if sequence.generated >= sequence.max_new_tokens {
            Some(SequenceFinishReason::MaxTokens)
        } else if sequence.position >= self.pipeline.config().max_positions {
            Some(SequenceFinishReason::Context)
        } else {
            None
        };
        if let Some(reason) = reason {
            self.finish(session, reason)?;
            return Ok(1);
        }
        let candidate = greedy_candidate(&LogitsOutput::Full(row.to_vec()));
        let Some(candidate) = candidate else {
            self.finish(session, SequenceFinishReason::NoCandidate)?;
            return Ok(1);
        };
        if self.stop_at_eos
            && !sequence.ignore_eos
            && self.tokenizer.is_eos_token(candidate.token_id)
        {
            self.finish(session, SequenceFinishReason::Eos)?;
            return Ok(1);
        }
        self.scheduler.stage_decode_candidate(
            session,
            candidate.token_id,
            Some(candidate.logit),
        )?;
        self.stats.staged_tokens += 1;
        Ok(0)
    }

    fn step_inner(
        &mut self,
        on_token: &mut dyn FnMut(&ResidentTokenEvent) -> Result<()>,
    ) -> Result<ResidentDriverStep> {
        self.available()?;
        self.flush(on_token)?;
        let Some(action) = self.scheduler.next_action(&mut self.slots)? else {
            if !self.requests.is_empty() {
                return Err(Error::Invariant {
                    message: "pipeline requests have no runnable action".into(),
                });
            }
            return Ok(ResidentDriverStep::Idle);
        };
        let staged_before = self.stats.staged_tokens;
        let (kind, rows, finished) = match action {
            SchedulerAction::PrefillChunk(action) => {
                let output = self.pipeline.forward(
                    action.session_id,
                    &action.tokens,
                    ForwardPhase::Prefill,
                )?;
                self.scheduler.commit_prefill_action(&action)?;
                self.stats.prefill_chunks += 1;
                self.stats.prefill_tokens += action.tokens.len();
                let finished = if action.logits == LogitsSelection::Last {
                    let row = output
                        .logits
                        .row(action.tokens.len() - 1)
                        .ok_or_else(|| invalid("pipeline omitted final prefill logits"))?;
                    self.next_candidate(action.session_id, row)?
                } else {
                    0
                };
                (ResidentActionKind::Prefill, action.tokens.len(), finished)
            }
            SchedulerAction::DecodeBatch(actions) if actions.len() == 1 => {
                let action = actions[0];
                let output = self.pipeline.forward(
                    action.session_id,
                    &[action.token_id],
                    ForwardPhase::Decode,
                )?;
                self.scheduler.commit_decode_action(&action)?;
                self.stats.decode_steps += 1;
                let sequence = self
                    .scheduler
                    .active_sequence_mut(action.session_id)
                    .ok_or_else(|| invalid("missing committed pipeline session"))?;
                let text = sequence
                    .incremental_decode
                    .step(action.token_id, |ids| self.tokenizer.decode(ids))?
                    .unwrap_or_default();
                sequence.append_generated_text(&text);
                let event = ResidentTokenEvent {
                    session_id: action.session_id,
                    request_id: action.request_id,
                    index: sequence.generated - 1,
                    token: action.token_id,
                    logit: action.logit,
                    text,
                };
                let stopped = self
                    .text
                    .get_mut(&action.session_id)
                    .expect("admission installs text policy")
                    .push(event, &sequence.stop, &mut self.outbox);
                let finished = if stopped {
                    self.finish(action.session_id, SequenceFinishReason::StopString)?;
                    1
                } else {
                    let row = output
                        .logits
                        .row(0)
                        .ok_or_else(|| invalid("pipeline omitted decode logits"))?;
                    self.next_candidate(action.session_id, row)?
                };
                (ResidentActionKind::Decode, 1, finished)
            }
            _ => {
                return Err(Error::Invariant {
                    message: "pipeline scheduler produced a non-serial action".into(),
                });
            }
        };
        self.stats.actions += 1;
        self.flush(on_token)?;
        Ok(ResidentDriverStep::Executed {
            action_kind: kind,
            rows,
            staged: self.stats.staged_tokens - staged_before,
            finished,
        })
    }
}

impl InferenceEngine for PipelineInferenceEngine {
    fn completion_hub(&self) -> CompletionHub {
        self.completion.clone()
    }
    fn take_completion_reactors(&mut self) -> Vec<InferenceCompletionReactor> {
        Vec::new()
    }
    fn has_pending_async_work(&self) -> bool {
        false
    }
    fn encode(&self, prompt: &str) -> Result<Vec<u32>> {
        Ok(self.tokenizer.encode(prompt)?)
    }

    fn request_cleanup(&self, request: RequestId) -> super::InferenceRequestCleanup {
        self.cleanup_receipts.get(&request).map_or(
            super::InferenceRequestCleanup::Unavailable,
            |(receipt, _)| super::InferenceRequestCleanup::Tracked(receipt.receipt()),
        )
    }

    fn try_submit(&mut self, mut request: GenerateRequest) -> Result<()> {
        self.available()?;
        let session = request.session_id.unwrap_or(SessionId(request.id.0));
        if self.requests.contains_key(&request.id)
            || self.cleanup_receipts.contains_key(&request.id)
            || self.pending_terminals.contains(&request.id)
            || self.requests.values().any(|&id| id == session)
        {
            return Err(invalid(
                "request or session already has an active turn or unconsumed terminal",
            ));
        }
        if request.prompt_tokens.is_empty()
            || request.max_new_tokens == 0
            || request
                .prompt_tokens
                .iter()
                .any(|&id| id as usize >= self.info.vocab_size)
            || request.stop.len() > 4
            || request
                .stop
                .iter()
                .any(|stop| stop.is_empty() || stop.len() > 4096)
        {
            return Err(invalid(
                "invalid pipeline prompt, token budget, vocabulary ID or stop strings",
            ));
        }
        let position = self.position(session).unwrap_or(0);
        if position
            .checked_add(request.prompt_tokens.len())
            .is_none_or(|end| end > self.pipeline.config().max_positions)
        {
            return Err(invalid("prompt exceeds the pipeline context capacity"));
        }
        if self.pipeline.session_slot(session).is_some() {
            if request.session_id.is_none() || !self.retained.contains(&session) {
                return Err(invalid(
                    "existing pipeline session was not explicitly retained",
                ));
            }
        } else {
            self.check_session_capacity()?;
            if let Err(error) = self.pipeline.create_session(session) {
                self.faulted = true;
                return Err(error.into());
            }
        }
        request.session_id = Some(session);
        self.requests.insert(request.id, session);
        self.cleanup_receipts
            .insert(request.id, (super::RequestCleanupOwner::default(), false));
        self.text.insert(session, StopText::default());
        self.scheduler.submit_at_position(request, position);
        Ok(())
    }

    fn submit(&mut self, request: GenerateRequest) {
        let rejected = request.clone();
        if let Err(error) = self.try_submit(request) {
            tracing::warn!(%error, "pipeline request rejected");
            // Legacy rejection may produce its own terminal, but must never
            // impersonate an existing request or replace an unconsumed outcome.
            if self.requests.contains_key(&rejected.id)
                || !self.pending_terminals.insert(rejected.id)
            {
                return;
            }
            let mut sequence = SequenceState::from_request(&rejected, SessionId(rejected.id.0));
            sequence.mark_error();
            self.rejected.push(sequence);
        }
    }

    fn step(
        &mut self,
        on_token: &mut dyn FnMut(&ResidentTokenEvent) -> Result<()>,
    ) -> Result<ResidentDriverStep> {
        let result = self.step_inner(on_token);
        self.reap_cleanup_receipts();
        if result.is_err() {
            self.faulted = true;
        }
        result
    }

    fn cancel_request(&mut self, request: RequestId) -> Result<InferenceCancelProgress> {
        if let Some(&session) = self.requests.get(&request) {
            // Cancelled turns are not retained, including their partially
            // executed prompt. Failed release keeps request and page custody.
            self.pipeline.release_session(session)?;
            self.retained.remove(&session);
            self.text.remove(&session);
            self.outbox
                .retain(|event| event.request_id != Some(request));
            self.requests.remove(&request);
        }
        let result = self.scheduler.cancel_request(request, &mut self.slots)?;
        if !matches!(result, crate::CancelRequestResult::NotFound { .. }) {
            self.pending_terminals.insert(request);
        }
        Ok(InferenceCancelProgress::Complete(result))
    }

    fn drain_finished(&mut self) -> Vec<SequenceState> {
        let sequences = self.scheduler.drain_finished();
        self.consume_terminals(&sequences);
        self.reap_cleanup_receipts();
        sequences
    }
    fn drain_cancelled(&mut self) -> Vec<SequenceState> {
        let sequences = self.scheduler.drain_cancelled();
        self.consume_terminals(&sequences);
        self.reap_cleanup_receipts();
        sequences
    }
    fn drain_failed(&mut self) -> Vec<SequenceState> {
        let mut sequences = self.scheduler.drain_failed();
        sequences.append(&mut self.rejected);
        self.consume_terminals(&sequences);
        self.reap_cleanup_receipts();
        sequences
    }

    fn shutdown(&mut self) -> Result<InferenceShutdownProgress> {
        self.closed = true;
        for request in self.requests.keys().copied().collect::<Vec<_>>() {
            self.cancel_request(request)?;
        }
        self.pipeline.shutdown()?;
        self.retained.clear();
        self.outbox.clear();
        Ok(InferenceShutdownProgress::Complete)
    }
}

impl SessionInferenceEngine for PipelineInferenceEngine {
    fn model_info(&self) -> ModelInfo {
        self.info.clone()
    }
    fn bound_layer_count(&self) -> Option<usize> {
        Some(self.info.num_layers)
    }
    fn expert_report(&self) -> Option<String> {
        None
    }
    fn observability_snapshot(&self) -> ResidentEngineObservability {
        ResidentEngineObservability {
            model: self.info.clone(),
            driver: self.stats.clone(),
            prefix_cache: Default::default(),
            materialization: Default::default(),
            kv_cache: Some(ResidentKvCacheObservability {
                page_size_tokens: self.pipeline.config().page_size,
                full_capacity_pages: self.kv_plan.full_capacity_pages,
                configured_pages: self.kv_plan.configured_pages,
                page_bytes: self.kv_plan.page_bytes,
                configured_bytes: self.kv_plan.configured_bytes,
                stats: self.pipeline.page_manager().stats(),
            }),
        }
    }
    fn retain_session(&mut self, session: SessionId) -> Result<()> {
        self.idle_session(session)?;
        if !self.retained.contains(&session) {
            self.check_session_capacity()?;
            self.pipeline.create_session(session)?;
            self.retained.insert(session);
        }
        Ok(())
    }
    fn retained_session_position(&self, session: SessionId) -> Option<usize> {
        self.retained
            .contains(&session)
            .then(|| self.position(session).unwrap_or(0))
    }
    fn reset_session(&mut self, session: SessionId) -> Result<()> {
        self.idle_session(session)?;
        if !self.retained.contains(&session) {
            return Err(invalid("session is not retained"));
        }
        if self.pipeline.session_slot(session).is_some() {
            self.pipeline.release_session(session)?;
        }
        self.pipeline.create_session(session)?;
        Ok(())
    }
    fn take_request_terminal(&mut self, request: RequestId) -> Option<RequestTerminal> {
        let terminal = if let Some(index) = self
            .rejected
            .iter()
            .position(|s| s.request_id == Some(request))
        {
            RequestTerminal::Failed(self.rejected.remove(index))
        } else {
            self.scheduler.take_request_terminal(request)?
        };
        self.pending_terminals.remove(&request);
        self.consume_cleanup_terminal(request);
        self.reap_cleanup_receipts();
        Some(terminal)
    }
}

/// At most one token event plus a potential stop prefix is held back. Empty
/// text events still count as committed tokens; terminal flush never invents a
/// token. A stop may span tokens or end in the middle of a tokenizer delta.
#[derive(Default)]
struct StopText {
    suffix: String,
    pending: Option<ResidentTokenEvent>,
}
impl StopText {
    fn push(
        &mut self,
        mut event: ResidentTokenEvent,
        stops: &[String],
        outbox: &mut VecDeque<ResidentTokenEvent>,
    ) -> bool {
        if let Some(previous) = self.pending.take() {
            outbox.push_back(previous);
        }
        self.suffix.push_str(&event.text);
        if let Some(end) = stops.iter().filter_map(|stop| self.suffix.find(stop)).min() {
            event.text = self.suffix[..end].to_owned();
            self.suffix.clear();
            outbox.push_back(event);
            return true;
        }
        let keep_from = self
            .suffix
            .char_indices()
            .map(|(index, _)| index)
            .find(|&index| {
                stops
                    .iter()
                    .any(|stop| stop.starts_with(&self.suffix[index..]))
            })
            .unwrap_or(self.suffix.len());
        event.text = self.suffix[..keep_from].to_owned();
        self.suffix = self.suffix[keep_from..].to_owned();
        if self.suffix.is_empty() {
            outbox.push_back(event);
        } else {
            self.pending = Some(event);
        }
        false
    }
    fn finish(&mut self, outbox: &mut VecDeque<ResidentTokenEvent>) {
        if let Some(mut event) = self.pending.take() {
            event.text.push_str(&self.suffix);
            outbox.push_back(event);
        }
        self.suffix.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn event(index: usize, text: &str) -> ResidentTokenEvent {
        ResidentTokenEvent {
            session_id: SessionId(1),
            request_id: Some(RequestId(1)),
            index,
            token: index as u32,
            logit: None,
            text: text.into(),
        }
    }
    #[test]
    fn stop_spans_tokens_and_discards_trailing_delta_without_losing_token_events() {
        let mut policy = StopText::default();
        let mut output = VecDeque::new();
        let stops = vec!["世界".into()];
        assert!(!policy.push(event(0, "hello世"), &stops, &mut output));
        assert!(output.is_empty());
        assert!(policy.push(event(1, "界extra"), &stops, &mut output));
        policy.finish(&mut output);
        assert_eq!(output.len(), 2);
        assert_eq!(
            output.iter().map(|e| e.text.as_str()).collect::<String>(),
            "hello"
        );
        assert_eq!(output[1].index, 1);
    }
    #[test]
    fn unfinished_stop_prefix_flushes_on_the_last_real_token() {
        let mut policy = StopText::default();
        let mut output = VecDeque::new();
        assert!(!policy.push(event(0, "abc"), &["abcd".into()], &mut output));
        policy.finish(&mut output);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].text, "abc");
    }
}
