//! Session and request types for the Ferrule Runtime.
//!
//! Provides explicit ownership of session state, token history, and
//! generation requests — the foundation for multi-session serving.

use std::ops::Range;

use ferrule_common::{Error, Result};

use crate::cache::KvHandle;

/// Unique identifier for a generation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestId(pub u64);

/// Unique identifier for a persistent session (multi-turn conversation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SessionId(pub u64);

/// Why a generation sequence finished.
///
/// This lives with session lifecycle rather than a concrete model or serving
/// transport. Generation-specific result types may re-export it under a narrower
/// name for compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SequenceFinishReason {
    MaxTokens,
    Eos,
    StopString,
    Context,
    NoCandidate,
    Cancelled,
}

impl SequenceFinishReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::MaxTokens => "max_tokens",
            Self::Eos => "eos",
            Self::StopString => "stop_string",
            Self::Context => "context",
            Self::NoCandidate => "no_candidate",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Status of a generation sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    /// Waiting to be scheduled for prefill.
    Pending,
    /// Currently being processed (prefill or decode).
    Running,
    /// Generation finished normally.
    Finished,
    /// Generation was cancelled.
    Cancelled,
    /// Generation encountered an error.
    Error,
}

/// A generation request sent to the runtime.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub id: RequestId,
    /// Optional session to continue (None = new session).
    pub session_id: Option<SessionId>,
    pub prompt_tokens: Vec<u32>,
    pub sampling: crate::sampling::sampler::SamplingConfig,
    pub max_new_tokens: usize,
    pub stop: Vec<String>,
}

/// The state of a single generation session.
#[derive(Debug, Clone)]
pub struct SequenceState {
    /// Request that currently owns this sequence turn, if any.
    pub request_id: Option<RequestId>,
    pub session_id: SessionId,
    /// Runtime KV allocation owned by this live sequence, if any.
    pub kv_handle: Option<KvHandle>,
    /// Logical execution position in the resident session/KV state.
    pub position: usize,
    /// All tokens tracked for this session (prompt + generated tokens returned to callers).
    pub tokens: Vec<u32>,
    /// Number of tokens generated since the last prompt.
    pub generated: usize,
    /// Current generation status.
    pub status: SequenceStatus,
    /// Structured reason for the last terminal state, if any.
    pub finish_reason: Option<SequenceFinishReason>,
    /// Sampling policy attached to the current request turn.
    pub sampling: crate::sampling::sampler::SamplingConfig,
    /// Generation budget attached to the current request turn.
    pub max_new_tokens: usize,
    /// Stop strings attached to the current request turn.
    pub stop: Vec<String>,
    /// Prompt token count for the current turn.
    pub prompt_len: usize,
    /// Number of current-turn prompt tokens already executed in prefill.
    pub prompt_cursor: usize,
    /// Token selected from the latest logits and waiting to be executed by the next decode batch.
    pub next_decode_token: Option<u32>,
    /// Logit associated with `next_decode_token`, when available.
    pub next_decode_logit: Option<f32>,
    /// Decoded text generated in the current request turn.
    pub generated_text: String,
    /// Total accumulated generated tokens.
    pub total_generated: usize,
}

impl SequenceState {
    pub fn new(session_id: SessionId) -> Self {
        Self {
            request_id: None,
            session_id,
            kv_handle: None,
            position: 0,
            tokens: Vec::new(),
            generated: 0,
            status: SequenceStatus::Pending,
            finish_reason: None,
            sampling: crate::sampling::sampler::SamplingConfig::default(),
            max_new_tokens: 0,
            stop: Vec::new(),
            prompt_len: 0,
            prompt_cursor: 0,
            next_decode_token: None,
            next_decode_logit: None,
            generated_text: String::new(),
            total_generated: 0,
        }
    }

    pub fn from_request(request: &GenerateRequest, fallback_session_id: SessionId) -> Self {
        let mut state = Self::new(request.session_id.unwrap_or(fallback_session_id));
        state.request_id = Some(request.id);
        state.apply_request_config(request);
        state.set_prompt(&request.prompt_tokens);
        state
    }

    pub fn apply_request_config(&mut self, request: &GenerateRequest) {
        self.sampling = request.sampling.clone();
        self.max_new_tokens = request.max_new_tokens;
        self.stop = request.stop.clone();
    }

    pub fn bind_kv(&mut self, handle: KvHandle) {
        self.kv_handle = Some(handle);
    }

    pub fn clear_kv(&mut self) {
        self.kv_handle = None;
    }

    pub fn current_prompt_tokens(&self) -> &[u32] {
        let end = self.tokens.len().saturating_sub(self.generated);
        let start = end.saturating_sub(self.prompt_len);
        &self.tokens[start..end]
    }

    pub fn remaining_prompt_tokens(&self) -> usize {
        self.prompt_len.saturating_sub(self.prompt_cursor)
    }

    pub fn prompt_prefill_done(&self) -> bool {
        self.remaining_prompt_tokens() == 0
    }

    pub fn next_prompt_chunk_range(&self, max_tokens: usize) -> Option<Range<usize>> {
        if max_tokens == 0 || self.prompt_prefill_done() {
            return None;
        }
        let start = self.prompt_cursor;
        let end = start.saturating_add(max_tokens).min(self.prompt_len);
        Some(start..end)
    }

    pub fn prompt_tokens_for_range(&self, range: Range<usize>) -> Result<&[u32]> {
        if range.start > range.end || range.end > self.prompt_len {
            return Err(Error::Internal(format!(
                "prompt chunk range {:?} exceeds prompt length {}",
                range, self.prompt_len
            )));
        }
        Ok(&self.current_prompt_tokens()[range])
    }

    pub fn commit_prefill_tokens(&mut self, token_count: usize) -> Result<()> {
        if token_count > self.remaining_prompt_tokens() {
            return Err(Error::Internal(format!(
                "cannot commit {token_count} prompt tokens with only {} remaining",
                self.remaining_prompt_tokens()
            )));
        }
        self.prompt_cursor += token_count;
        self.position = self.position.saturating_add(token_count);
        self.status = SequenceStatus::Running;
        Ok(())
    }

    pub fn advance_position(&mut self, token_count: usize) {
        self.position = self.position.saturating_add(token_count);
    }

    pub fn stage_decode_token(&mut self, token_id: u32) -> Result<()> {
        self.stage_decode_candidate(token_id, None)
    }

    pub fn stage_decode_candidate(&mut self, token_id: u32, logit: Option<f32>) -> Result<()> {
        if !self.prompt_prefill_done() {
            return Err(Error::Internal(format!(
                "cannot stage decode token before prompt prefill completes: {}/{} prompt tokens committed",
                self.prompt_cursor, self.prompt_len
            )));
        }
        self.next_decode_token = Some(token_id);
        self.next_decode_logit = logit;
        Ok(())
    }

    pub fn commit_staged_decode_token(&mut self, token_id: u32) -> Result<()> {
        match self.next_decode_token {
            Some(expected) if expected == token_id => {
                self.next_decode_token = None;
                self.next_decode_logit = None;
                self.extend_generated(&[token_id]);
                Ok(())
            }
            Some(expected) => Err(Error::Internal(format!(
                "decode token mismatch: staged {expected}, committed {token_id}"
            ))),
            None => Err(Error::Internal(
                "cannot commit decode token without a staged token".into(),
            )),
        }
    }

    /// Append generated tokens to the session.
    pub fn extend_generated(&mut self, tokens: &[u32]) {
        self.tokens.extend_from_slice(tokens);
        self.generated += tokens.len();
        self.total_generated += tokens.len();
        self.position = self.position.saturating_add(tokens.len());
    }

    pub fn append_generated_text(&mut self, text: &str) {
        self.generated_text.push_str(text);
    }

    /// Record the prompt tokens for a one-shot request, replacing previous tokens.
    pub fn set_prompt(&mut self, tokens: &[u32]) {
        self.tokens.clear();
        self.tokens.extend_from_slice(tokens);
        self.position = 0;
        self.prompt_len = tokens.len();
        self.prompt_cursor = 0;
        self.next_decode_token = None;
        self.next_decode_logit = None;
        self.generated_text.clear();
        self.generated = 0;
        self.status = SequenceStatus::Running;
        self.finish_reason = None;
    }

    /// Append prompt tokens to a resident multi-turn session.
    ///
    /// This is the session metadata counterpart of SGLang/vLLM-style resident
    /// workers: prompt turns extend the live sequence instead of replacing it.
    pub fn append_prompt(&mut self, tokens: &[u32]) {
        self.tokens.extend_from_slice(tokens);
        self.prompt_len = tokens.len();
        self.prompt_cursor = 0;
        self.next_decode_token = None;
        self.next_decode_logit = None;
        self.generated_text.clear();
        self.generated = 0;
        self.status = SequenceStatus::Running;
        self.finish_reason = None;
    }

    /// Mark the current turn as finished with a structured terminal reason.
    pub fn mark_finished(&mut self, reason: SequenceFinishReason) {
        self.status = match reason {
            SequenceFinishReason::Cancelled => SequenceStatus::Cancelled,
            _ => SequenceStatus::Finished,
        };
        self.next_decode_token = None;
        self.next_decode_logit = None;
        self.finish_reason = Some(reason);
    }

    /// Mark the current turn as cancelled without rolling back tokens/KV state.
    pub fn mark_cancelled(&mut self) {
        self.mark_finished(SequenceFinishReason::Cancelled);
    }

    /// Reset the entire session.
    pub fn reset(&mut self) {
        self.request_id = None;
        self.clear_kv();
        self.position = 0;
        self.tokens.clear();
        self.generated = 0;
        self.prompt_len = 0;
        self.prompt_cursor = 0;
        self.next_decode_token = None;
        self.next_decode_logit = None;
        self.generated_text.clear();
        self.status = SequenceStatus::Pending;
        self.finish_reason = None;
        self.sampling = crate::sampling::sampler::SamplingConfig::default();
        self.max_new_tokens = 0;
        self.stop.clear();
        self.total_generated = 0;
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::KvHandle;

    #[test]
    fn request_id_equality() {
        let a = RequestId(1);
        let b = RequestId(1);
        let c = RequestId(2);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn session_lifecycle() {
        let mut seq = SequenceState::new(SessionId(42));
        assert_eq!(seq.session_id, SessionId(42));
        assert_eq!(seq.status, SequenceStatus::Pending);
        assert_eq!(seq.finish_reason, None);
        assert!(seq.tokens.is_empty());

        seq.set_prompt(&[1, 2, 3]);
        assert_eq!(seq.status, SequenceStatus::Running);
        assert_eq!(seq.finish_reason, None);
        assert_eq!(seq.prompt_len, 3);
        assert_eq!(seq.prompt_cursor, 0);
        assert_eq!(seq.position, 0);
        assert_eq!(seq.tokens, vec![1, 2, 3]);
        assert_eq!(seq.generated, 0);

        assert_eq!(seq.current_prompt_tokens(), &[1, 2, 3]);
        assert_eq!(seq.next_prompt_chunk_range(2), Some(0..2));
        assert_eq!(seq.prompt_tokens_for_range(0..2).unwrap(), &[1, 2]);
        seq.commit_prefill_tokens(3).unwrap();
        assert!(seq.prompt_prefill_done());
        assert_eq!(seq.position, 3);

        seq.extend_generated(&[4, 5]);
        assert_eq!(seq.tokens, vec![1, 2, 3, 4, 5]);
        assert_eq!(seq.generated, 2);
        assert_eq!(seq.total_generated, 2);
        assert_eq!(seq.position, 5);

        seq.extend_generated(&[6]);
        assert_eq!(seq.total_generated, 3);

        seq.mark_finished(SequenceFinishReason::MaxTokens);
        assert_eq!(seq.status, SequenceStatus::Finished);
        assert_eq!(seq.finish_reason, Some(SequenceFinishReason::MaxTokens));

        seq.append_prompt(&[7, 8]);
        assert_eq!(seq.status, SequenceStatus::Running);
        assert_eq!(seq.finish_reason, None);
        assert_eq!(seq.tokens, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(seq.current_prompt_tokens(), &[7, 8]);
        assert_eq!(seq.prompt_len, 2);
        assert_eq!(seq.prompt_cursor, 0);
        assert_eq!(seq.generated, 0);
        assert_eq!(seq.total_generated, 3);

        seq.reset();
        assert!(seq.tokens.is_empty());
        assert_eq!(seq.generated, 0);
        assert_eq!(seq.prompt_len, 0);
        assert_eq!(seq.prompt_cursor, 0);
        assert_eq!(seq.position, 0);
        assert_eq!(seq.status, SequenceStatus::Pending);
        assert_eq!(seq.finish_reason, None);
        assert_eq!(seq.max_new_tokens, 0);
        assert!(seq.stop.is_empty());
        assert_eq!(seq.total_generated, 0);
    }

    #[test]
    fn cancellation_sets_cancelled_status_and_reason() {
        let mut seq = SequenceState::new(SessionId(7));
        seq.set_prompt(&[1, 2, 3]);
        seq.extend_generated(&[4]);

        seq.mark_cancelled();

        assert_eq!(seq.status, SequenceStatus::Cancelled);
        assert_eq!(seq.finish_reason, Some(SequenceFinishReason::Cancelled));
        assert_eq!(seq.tokens, vec![1, 2, 3, 4]);
    }

    #[test]
    fn generate_request_fields() {
        let req = GenerateRequest {
            id: RequestId(100),
            session_id: Some(SessionId(1)),
            prompt_tokens: vec![10, 20],
            sampling: crate::sampling::sampler::SamplingConfig::greedy(),
            max_new_tokens: 32,
            stop: vec!["</s>".into()],
        };
        assert_eq!(req.id, RequestId(100));
        assert_eq!(req.session_id, Some(SessionId(1)));
        assert_eq!(req.max_new_tokens, 32);
        assert_eq!(req.stop, vec!["</s>"]);

        let state = SequenceState::from_request(&req, SessionId(99));
        assert_eq!(state.request_id, Some(RequestId(100)));
        assert_eq!(state.session_id, SessionId(1));
        assert_eq!(state.current_prompt_tokens(), &[10, 20]);
        assert_eq!(state.max_new_tokens, 32);
        assert_eq!(state.stop, vec!["</s>"]);
        assert_eq!(state.sampling.temperature, 0.0);
    }

    #[test]
    fn kv_binding_is_explicit_session_state() {
        let mut seq = SequenceState::new(SessionId(3));
        assert_eq!(seq.kv_handle, None);
        seq.bind_kv(KvHandle(8));
        assert_eq!(seq.kv_handle, Some(KvHandle(8)));
        seq.clear_kv();
        assert_eq!(seq.kv_handle, None);
    }

    #[test]
    fn staged_decode_token_requires_finished_prefill_and_commits_generation() {
        let mut seq = SequenceState::new(SessionId(4));
        seq.set_prompt(&[1, 2]);
        assert!(seq.stage_decode_token(9).is_err());

        seq.commit_prefill_tokens(2).unwrap();
        seq.stage_decode_token(9).unwrap();
        assert_eq!(seq.next_decode_token, Some(9));
        assert!(seq.commit_staged_decode_token(10).is_err());
        seq.commit_staged_decode_token(9).unwrap();

        assert_eq!(seq.next_decode_token, None);
        assert_eq!(seq.generated, 1);
        assert_eq!(seq.total_generated, 1);
        assert_eq!(seq.position, 3);
        assert_eq!(seq.tokens, vec![1, 2, 9]);
    }

    #[test]
    fn multiple_sessions_independent() {
        let mut s1 = SequenceState::new(SessionId(1));
        let mut s2 = SequenceState::new(SessionId(2));

        s1.set_prompt(&[1]);
        s2.set_prompt(&[100]);

        s1.extend_generated(&[2]);
        s2.extend_generated(&[200, 201]);

        assert_eq!(s1.tokens, vec![1, 2]);
        assert_eq!(s2.tokens, vec![100, 200, 201]);
        assert_eq!(s1.generated, 1);
        assert_eq!(s2.generated, 2);
    }
}
