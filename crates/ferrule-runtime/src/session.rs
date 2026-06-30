//! Session and request types for the Ferrule Runtime.
//!
//! Provides explicit ownership of session state, token history, and
//! generation requests — the foundation for multi-session serving.

/// Unique identifier for a generation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(pub u64);

/// Unique identifier for a persistent session (multi-turn conversation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(pub u64);

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
    pub sampling: crate::sampler::SamplingConfig,
    pub max_new_tokens: usize,
    pub stop: Vec<String>,
}

/// The state of a single generation session.
#[derive(Debug, Clone)]
pub struct SequenceState {
    pub session_id: SessionId,
    /// All tokens in this session (prompt + generated).
    pub tokens: Vec<u32>,
    /// Number of tokens generated since the last prompt.
    pub generated: usize,
    /// Current generation status.
    pub status: SequenceStatus,
    /// Prompt token count (for stats).
    pub prompt_len: usize,
    /// Total accumulated generated tokens.
    pub total_generated: usize,
}

impl SequenceState {
    pub fn new(session_id: SessionId) -> Self {
        Self {
            session_id,
            tokens: Vec::new(),
            generated: 0,
            status: SequenceStatus::Pending,
            prompt_len: 0,
            total_generated: 0,
        }
    }

    /// Append generated tokens to the session.
    pub fn extend_generated(&mut self, tokens: &[u32]) {
        self.tokens.extend_from_slice(tokens);
        self.generated += tokens.len();
        self.total_generated += tokens.len();
    }

    /// Record the prompt tokens.
    pub fn set_prompt(&mut self, tokens: &[u32]) {
        self.tokens.clear();
        self.tokens.extend_from_slice(tokens);
        self.prompt_len = tokens.len();
        self.generated = 0;
        self.status = SequenceStatus::Running;
    }

    /// Reset the entire session.
    pub fn reset(&mut self) {
        self.tokens.clear();
        self.generated = 0;
        self.prompt_len = 0;
        self.status = SequenceStatus::Pending;
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(seq.tokens.is_empty());

        seq.set_prompt(&[1, 2, 3]);
        assert_eq!(seq.status, SequenceStatus::Running);
        assert_eq!(seq.prompt_len, 3);
        assert_eq!(seq.tokens, vec![1, 2, 3]);
        assert_eq!(seq.generated, 0);

        seq.extend_generated(&[4, 5]);
        assert_eq!(seq.tokens, vec![1, 2, 3, 4, 5]);
        assert_eq!(seq.generated, 2);
        assert_eq!(seq.total_generated, 2);

        seq.extend_generated(&[6]);
        assert_eq!(seq.total_generated, 3);

        seq.reset();
        assert!(seq.tokens.is_empty());
        assert_eq!(seq.generated, 0);
        assert_eq!(seq.prompt_len, 0);
        assert_eq!(seq.status, SequenceStatus::Pending);
        // total_generated is NOT reset — it's a cumulative counter
        assert_eq!(seq.total_generated, 3);
    }

    #[test]
    fn generate_request_fields() {
        let req = GenerateRequest {
            id: RequestId(100),
            session_id: Some(SessionId(1)),
            prompt_tokens: vec![10, 20],
            sampling: crate::sampler::SamplingConfig::greedy(),
            max_new_tokens: 32,
            stop: vec!["</s>".into()],
        };
        assert_eq!(req.id, RequestId(100));
        assert_eq!(req.session_id, Some(SessionId(1)));
        assert_eq!(req.max_new_tokens, 32);
        assert_eq!(req.stop, vec!["</s>"]);
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
