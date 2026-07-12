use std::time::{Duration, Instant};

use ferrule_common::{Error, Result};
use ferrule_model::{PrefillMode, TokenLogit, TopKModelRunner};

use crate::generation::{
    ensure_context_room, matched_stop, GenerationConfig, TopKFinishReason, TopKTokenEvent,
    TopKTurnResult,
};
use crate::scheduling::{SequenceState, SessionId};

/// Snapshot of a resident worker's generic runtime state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineWorkerStats {
    pub session_id: SessionId,
    /// Runner-reported absolute position in its resident session/KV state.
    pub position: usize,
    /// Tokens tracked by runtime session metadata.
    pub tracked_tokens: usize,
    /// Number of turns completed by this worker since the last reset.
    pub turns: u64,
    /// Prompt tokens appended since the last reset.
    pub prompt_tokens: usize,
    /// Generated tokens returned to callers since the last reset.
    pub generated_tokens: usize,
    /// Optional number of materialized/bound execution layers.
    pub bound_layers: Option<usize>,
}

/// State produced by `EngineWorker::append_prompt` and consumed by
/// `EngineWorker::decode_next`.
///
/// This is intentionally a concrete Rust value instead of a trait object: a
/// scheduler can hold decode state explicitly, and the compiler keeps prompt
/// prefill and decode progression separated without introducing a large engine
/// framework prematurely.
#[must_use = "decode state must be driven with EngineWorker::decode_next or explicitly finished"]
pub struct TopKDecodeState {
    prompt_tokens: usize,
    prefill_time: Duration,
    decode_start: Instant,
    config: GenerationConfig,
    top_k: usize,
    current_topk: Vec<TokenLogit>,
    tokens: Vec<u32>,
    text: String,
    next_index: usize,
    stopped_by_eos: bool,
    stopped_by_string: Option<String>,
    stopped_by_context: bool,
    finish_reason: Option<TopKFinishReason>,
    finished: bool,
    accounted: bool,
}

impl TopKDecodeState {
    fn new(
        prompt_tokens: usize,
        prefill_time: Duration,
        config: GenerationConfig,
        top_k: usize,
        current_topk: Vec<TokenLogit>,
    ) -> Self {
        Self {
            prompt_tokens,
            prefill_time,
            decode_start: Instant::now(),
            config,
            top_k,
            current_topk,
            tokens: Vec::new(),
            text: String::new(),
            next_index: 0,
            stopped_by_eos: false,
            stopped_by_string: None,
            stopped_by_context: false,
            finish_reason: None,
            finished: false,
            accounted: false,
        }
    }

    pub fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    pub fn generated_tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn generated_text(&self) -> &str {
        &self.text
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn finish_reason(&self) -> Option<TopKFinishReason> {
        self.finish_reason
    }
}

/// One step of incremental top-k decode.
#[must_use = "decode steps must be handled to stream tokens or observe turn completion"]
#[derive(Debug, Clone)]
pub enum TopKDecodeStep {
    /// A token was committed to the resident session and should be streamed to
    /// the caller.
    Token(TopKTokenEvent),
    /// The turn is complete and all timing/stop metadata is available.
    Finished(TopKTurnResult),
}

/// Long-lived single-runner worker.
///
/// The worker is generic over `TopKModelRunner`: concrete model loading stays in
/// `ferrule-model`/CLI, while runtime owns session accounting, context-aware
/// turn execution, reset semantics, and worker stats.
pub struct EngineWorker<R: TopKModelRunner> {
    runner: R,
    session: SequenceState,
    turns: u64,
    prompt_tokens: usize,
}

impl<R: TopKModelRunner> EngineWorker<R> {
    pub fn new(runner: R) -> Self {
        Self::with_session(runner, SessionId(0))
    }

    pub fn with_session(runner: R, session_id: SessionId) -> Self {
        Self {
            runner,
            session: SequenceState::new(session_id),
            turns: 0,
            prompt_tokens: 0,
        }
    }

    pub fn runner(&self) -> &R {
        &self.runner
    }

    pub fn runner_mut(&mut self) -> &mut R {
        &mut self.runner
    }

    pub fn session(&self) -> &SequenceState {
        &self.session
    }

    pub fn position(&self) -> usize {
        self.runner.position()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.runner.encode(text)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.runner.reset_session()?;
        self.session.reset();
        self.turns = 0;
        self.prompt_tokens = 0;
        Ok(())
    }

    pub fn expert_report(&self) -> Option<String> {
        self.runner.expert_report()
    }

    pub fn stats(&self) -> EngineWorkerStats {
        EngineWorkerStats {
            session_id: self.session.session_id,
            position: self.runner.position(),
            tracked_tokens: self.session.tokens.len(),
            turns: self.turns,
            prompt_tokens: self.prompt_tokens,
            generated_tokens: self.session.total_generated,
            bound_layers: self.runner.bound_layer_count(),
        }
    }

    /// Append a prompt turn to the resident session and produce decode state.
    ///
    /// This is the explicit prefill/admission phase. It is still single-worker
    /// and synchronous, but it gives the scheduler a real state object to hold
    /// before repeatedly calling `decode_next`.
    pub fn append_prompt(
        &mut self,
        prompt_tokens: &[u32],
        config: &GenerationConfig,
        prefill_mode: PrefillMode,
        top_k: usize,
    ) -> Result<TopKDecodeState> {
        if top_k == 0 {
            return Err(Error::Internal("top_k must be greater than zero".into()));
        }
        ensure_context_room(self.runner.position(), prompt_tokens.len(), config.ctx_size)?;

        let prefill_start = Instant::now();
        let current_topk = self
            .runner
            .prefill_topk(prompt_tokens, top_k, prefill_mode)?;
        let prefill_time = prefill_start.elapsed();

        self.session.append_prompt(prompt_tokens);
        self.session.commit_prefill_tokens(prompt_tokens.len())?;
        self.prompt_tokens = self.prompt_tokens.saturating_add(prompt_tokens.len());

        Ok(TopKDecodeState::new(
            prompt_tokens.len(),
            prefill_time,
            config.clone(),
            top_k,
            current_topk,
        ))
    }

    /// Advance one decode step for an active top-k turn.
    pub fn decode_next(&mut self, state: &mut TopKDecodeState) -> Result<TopKDecodeStep> {
        if state.finished {
            return Ok(TopKDecodeStep::Finished(self.finish_decode_state(state)));
        }
        if state.config.ctx_size == 0 {
            return Err(Error::Internal("ctx_size must be greater than zero".into()));
        }
        if state.next_index >= state.config.max_new_tokens {
            state.finish_reason = Some(TopKFinishReason::MaxTokens);
            state.finished = true;
            return Ok(TopKDecodeStep::Finished(self.finish_decode_state(state)));
        }
        if self.runner.position() >= state.config.ctx_size {
            state.stopped_by_context = true;
            state.finish_reason = Some(TopKFinishReason::Context);
            state.finished = true;
            return Ok(TopKDecodeStep::Finished(self.finish_decode_state(state)));
        }
        let Some(next) = state.current_topk.first().copied() else {
            state.finish_reason = Some(TopKFinishReason::NoCandidate);
            state.finished = true;
            return Ok(TopKDecodeStep::Finished(self.finish_decode_state(state)));
        };

        if state.config.stop_at_eos && self.runner.eos_token_id() == Some(next.token_id) {
            state.stopped_by_eos = true;
            state.finish_reason = Some(TopKFinishReason::Eos);
            if state.config.append_eos_to_session {
                self.runner.feed_token(next.token_id)?;
                self.session.advance_position(1);
            }
            state.finished = true;
            return Ok(TopKDecodeStep::Finished(self.finish_decode_state(state)));
        }

        let piece = self.runner.decode(&[next.token_id])?;
        let event = TopKTokenEvent {
            index: state.next_index,
            token: next.token_id,
            logit: next.logit,
            text: piece.clone(),
        };

        state.tokens.push(next.token_id);
        state.text.push_str(&piece);
        state.next_index += 1;
        self.session.extend_generated(&[next.token_id]);

        if let Some(stop) = matched_stop(&state.text, &state.config.stop) {
            self.runner.feed_token(next.token_id)?;
            state.stopped_by_string = Some(stop);
            state.finish_reason = Some(TopKFinishReason::StopString);
            state.finished = true;
            return Ok(TopKDecodeStep::Token(event));
        }

        if state.next_index >= state.config.max_new_tokens {
            self.runner.feed_token(next.token_id)?;
            state.finish_reason = Some(TopKFinishReason::MaxTokens);
            state.finished = true;
            return Ok(TopKDecodeStep::Token(event));
        }

        state.current_topk = self.runner.decode_topk(next.token_id, state.top_k)?;
        Ok(TopKDecodeStep::Token(event))
    }

    pub fn cancel_decode(&mut self, state: &mut TopKDecodeState) -> TopKTurnResult {
        if state.finished {
            state
                .finish_reason
                .get_or_insert(TopKFinishReason::MaxTokens);
        } else {
            state.finish_reason = Some(TopKFinishReason::Cancelled);
            state.finished = true;
        }
        self.finish_decode_state(state)
    }

    fn finish_decode_state(&mut self, state: &mut TopKDecodeState) -> TopKTurnResult {
        if !state.accounted {
            self.turns = self.turns.saturating_add(1);
            state.accounted = true;
        }
        let finish_reason = state.finish_reason.unwrap_or(TopKFinishReason::MaxTokens);
        self.session.mark_finished(finish_reason);

        TopKTurnResult {
            prompt_tokens: state.prompt_tokens,
            tokens: state.tokens.clone(),
            text: state.text.clone(),
            prefill_time: state.prefill_time,
            decode_time: state.decode_start.elapsed(),
            stopped_by_eos: state.stopped_by_eos,
            stopped_by_string: state.stopped_by_string.clone(),
            stopped_by_context: state.stopped_by_context,
            finish_reason,
            final_position: self.runner.position(),
        }
    }
}
