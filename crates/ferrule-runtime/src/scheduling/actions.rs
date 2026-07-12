//! Executable scheduling action vocabulary and prefill planning.
//!
//! These actions carry resident sequence work into runtime-private execution
//! lowering without owning queueing, admission, or KV lifecycle policy.

use crate::cache::KvHandle;
use ferrule_common::{Error, Result};

use super::session::{RequestId, SequenceFinishReason, SequenceState, SessionId};

/// Maximum tokens to process in one prefill chunk.
/// Larger = higher throughput but worse latency for concurrent decodes.
pub const DEFAULT_CHUNK_SIZE: usize = 256;

/// Logits rows requested while scheduling a prefill chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogitsSelection {
    None,
    Last,
    All,
}

/// A concrete scheduler decision that can be executed by a resident worker or a
/// graph-backed runner.
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulerAction {
    PrefillChunk(PrefillChunkAction),
    DecodeBatch(Vec<DecodeAction>),
    Finish {
        request_id: Option<RequestId>,
        session_id: SessionId,
        reason: SequenceFinishReason,
    },
    Cancel {
        request_id: Option<RequestId>,
        session_id: SessionId,
    },
}

/// One chunk of prompt prefill for a resident sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillChunkAction {
    pub request_id: Option<RequestId>,
    pub session_id: SessionId,
    /// Relative range within the current prompt turn.
    pub token_range: std::ops::Range<usize>,
    /// Absolute logical position for the first token in `tokens`.
    pub position_start: usize,
    pub tokens: Vec<u32>,
    pub kv_handle: Option<KvHandle>,
    pub logits: LogitsSelection,
}

impl PrefillChunkAction {
    pub fn from_sequence(sequence: &SequenceState, max_tokens: usize) -> Result<Option<Self>> {
        let Some(range) = sequence.next_prompt_chunk_range(max_tokens) else {
            return Ok(None);
        };
        let tokens = sequence.prompt_tokens_for_range(range.clone())?.to_vec();
        let logits = if range.end == sequence.prompt_len {
            LogitsSelection::Last
        } else {
            LogitsSelection::None
        };
        Ok(Some(Self {
            request_id: sequence.request_id,
            session_id: sequence.session_id,
            token_range: range,
            position_start: sequence.position,
            tokens,
            kv_handle: sequence.kv_handle,
            logits,
        }))
    }

    pub fn commit(&self, sequence: &mut SequenceState) -> Result<()> {
        if sequence.session_id != self.session_id {
            return Err(Error::Internal(format!(
                "cannot commit prefill for session {:?} into session {:?}",
                self.session_id, sequence.session_id
            )));
        }
        if self.token_range.end < self.token_range.start {
            return Err(Error::Internal(format!(
                "cannot commit reversed prefill token range {:?}",
                self.token_range
            )));
        }
        if sequence.prompt_cursor != self.token_range.start {
            return Err(Error::Internal(format!(
                "prefill cursor mismatch: sequence at {}, action starts at {}",
                sequence.prompt_cursor, self.token_range.start
            )));
        }
        if sequence.position != self.position_start {
            return Err(Error::Internal(format!(
                "prefill position mismatch: sequence at {}, action starts at {}",
                sequence.position, self.position_start
            )));
        }
        sequence.commit_prefill_tokens(self.token_range.len())
    }
}

/// One row in a decode batch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeAction {
    pub request_id: Option<RequestId>,
    pub session_id: SessionId,
    pub token_id: u32,
    pub logit: Option<f32>,
    pub position: usize,
    pub kv_handle: Option<KvHandle>,
    /// Whether the backend should return logits for the next token after feeding
    /// `token_id`. The last token before `max_new_tokens` can skip this work.
    pub require_logits: bool,
}

impl DecodeAction {
    pub fn from_sequence(sequence: &SequenceState, token_id: u32) -> Self {
        Self {
            request_id: sequence.request_id,
            session_id: sequence.session_id,
            token_id,
            logit: sequence.next_decode_logit,
            position: sequence.position,
            kv_handle: sequence.kv_handle,
            require_logits: sequence.generated.saturating_add(1) < sequence.max_new_tokens,
        }
    }
}

pub fn plan_prefill_chunk(
    sequence: &SequenceState,
    max_tokens: usize,
) -> Result<Option<SchedulerAction>> {
    PrefillChunkAction::from_sequence(sequence, max_tokens)
        .map(|action| action.map(SchedulerAction::PrefillChunk))
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use crate::scheduling::ScheduledBatch;

    #[test]
    fn prefill_action_moves_payload_and_commits_by_range() {
        let mut sequence = SequenceState::new(SessionId(5));
        sequence.request_id = Some(RequestId(9));
        sequence.bind_kv(KvHandle(2));
        sequence.set_prompt(&[10, 11, 12]);

        let mut first_action = plan_prefill_chunk(&sequence, 2).unwrap().unwrap();
        let first_batch =
            ScheduledBatch::from_action(&mut first_action, NonZeroU32::new(1).unwrap())
                .unwrap()
                .unwrap();
        let SchedulerAction::PrefillChunk(first) = &first_action else {
            panic!("expected prefill action");
        };
        assert_eq!(first.request_id, Some(RequestId(9)));
        assert_eq!(first.token_range, 0..2);
        assert!(first.tokens.is_empty());
        assert_eq!(first_batch.execution().token_ids(), &[10, 11]);
        assert_eq!(first.position_start, 0);
        assert_eq!(first.logits, LogitsSelection::None);
        assert_eq!(first.kv_handle, Some(KvHandle(2)));

        first.commit(&mut sequence).unwrap();
        assert_eq!(sequence.prompt_cursor, 2);
        assert_eq!(sequence.position, 2);

        let mut last_action = plan_prefill_chunk(&sequence, 4).unwrap().unwrap();
        let last_batch = ScheduledBatch::from_action(&mut last_action, NonZeroU32::new(1).unwrap())
            .unwrap()
            .unwrap();
        let SchedulerAction::PrefillChunk(last) = &last_action else {
            panic!("expected final prefill action");
        };
        assert_eq!(last.token_range, 2..3);
        assert!(last.tokens.is_empty());
        assert_eq!(last_batch.execution().token_ids(), &[12]);
        assert_eq!(last.position_start, 2);
        assert_eq!(last.logits, LogitsSelection::Last);

        last.commit(&mut sequence).unwrap();
        assert!(sequence.prompt_prefill_done());
        assert_eq!(sequence.position, 3);
        assert!(plan_prefill_chunk(&sequence, 1).unwrap().is_none());
    }

    #[test]
    fn decode_action_captures_execution_inputs() {
        let mut sequence = SequenceState::new(SessionId(6));
        sequence.request_id = Some(RequestId(12));
        sequence.bind_kv(KvHandle(4));
        sequence.max_new_tokens = 2;
        sequence.set_prompt(&[1, 2, 3]);
        sequence.commit_prefill_tokens(3).unwrap();

        let action = DecodeAction::from_sequence(&sequence, 42);
        assert_eq!(action.request_id, Some(RequestId(12)));
        assert_eq!(action.position, 3);
        assert_eq!(action.token_id, 42);
        assert_eq!(action.kv_handle, Some(KvHandle(4)));
        assert!(action.require_logits);
    }
}
