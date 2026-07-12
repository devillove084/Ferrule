//! Runtime-private lowering from scheduler actions to the neutral execution ABI.

use std::num::NonZeroU32;

use ferrule_common::execution::{
    ExecutionBatch, ExecutionOutput, ExecutionSequence, ForwardMode, ForwardPhase, LogitsRequest,
    StateSlot,
};
use ferrule_common::{Error, Result};

use crate::cache::KvHandle;

use super::actions::{LogitsSelection, SchedulerAction};
use super::session::{RequestId, SessionId};

/// Runtime correlation for one sequence in a neutral execution batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScheduledSequence {
    pub(crate) state_slot: StateSlot,
    pub(crate) request_id: Option<RequestId>,
    pub(crate) session_id: SessionId,
    pub(crate) kv_handle: Option<KvHandle>,
}

/// A neutral execution batch paired with runtime-owned sequence correlation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ScheduledBatch {
    pub(crate) execution: ExecutionBatch,
    pub(crate) sequences: Vec<ScheduledSequence>,
}

impl ScheduledBatch {
    /// Lowers an executable scheduler action into the dependency-neutral ABI.
    /// Terminal actions and empty decode batches do not require model execution.
    pub(crate) fn from_action(
        action: &mut SchedulerAction,
        top_k: NonZeroU32,
    ) -> Result<Option<Self>> {
        match action {
            SchedulerAction::PrefillChunk(action) => Self::from_prefill(action, top_k).map(Some),
            SchedulerAction::DecodeBatch(actions) if actions.is_empty() => Ok(None),
            SchedulerAction::DecodeBatch(actions) => {
                let batch = Self::from_decode(actions, top_k)?;
                batch.validate_correlation()?;
                Ok(Some(batch))
            }
            SchedulerAction::Finish { .. } | SchedulerAction::Cancel { .. } => Ok(None),
        }
    }

    pub(crate) fn execution(&self) -> &ExecutionBatch {
        &self.execution
    }

    #[cfg(test)]
    pub(crate) fn sequences(&self) -> &[ScheduledSequence] {
        &self.sequences
    }

    #[cfg(test)]
    pub(crate) fn sequence_for_state_slot(
        &self,
        state_slot: StateSlot,
    ) -> Option<&ScheduledSequence> {
        self.sequences
            .iter()
            .find(|sequence| sequence.state_slot == state_slot)
    }

    /// Resolves a packed input row through its execution span to runtime state.
    pub(crate) fn sequence_for_input_row(&self, input_row: u32) -> Option<&ScheduledSequence> {
        Self::sequence_for_input_row_parts(self.execution.sequences(), &self.sequences, input_row)
    }

    fn sequence_for_input_row_parts<'a>(
        execution_sequences: &[ExecutionSequence],
        scheduled_sequences: &'a [ScheduledSequence],
        input_row: u32,
    ) -> Option<&'a ScheduledSequence> {
        let mut matching_sequences = execution_sequences
            .iter()
            .filter(|sequence| sequence.query.contains(&input_row));
        let execution_sequence = matching_sequences.next()?;
        if matching_sequences.next().is_some() {
            return None;
        }
        scheduled_sequences
            .iter()
            .find(|sequence| sequence.state_slot == execution_sequence.state_slot)
    }

    /// Validates neutral output shape and then its runtime correlation.
    pub(crate) fn validate_output(&self, output: &ExecutionOutput) -> Result<()> {
        output.validate(&self.execution)?;
        self.validate_correlation()?;

        for row in &output.logits {
            if self.sequence_for_input_row(row.input_row).is_none() {
                return Err(execution_error(format!(
                    "output input row {} has no unique scheduled-sequence correlation",
                    row.input_row
                )));
            }
        }

        Ok(())
    }

    fn from_prefill(
        action: &mut super::actions::PrefillChunkAction,
        top_k: NonZeroU32,
    ) -> Result<Self> {
        if action.token_range.end < action.token_range.start {
            return Err(execution_error(format!(
                "prefill token range {:?} is reversed",
                action.token_range
            )));
        }

        let token_count = action.tokens.len();
        let range_len = action.token_range.len();
        if range_len != token_count {
            return Err(execution_error(format!(
                "prefill token range {:?} has length {range_len}, but the payload has {token_count} tokens",
                action.token_range
            )));
        }
        if token_count == 0 {
            return Err(execution_error(
                "cannot lower an empty prefill action into an execution batch",
            ));
        }

        let query_len = checked_u32(token_count, "prefill token count")?;
        let context_len = checked_u32(action.position_start, "prefill position")?;
        let sequence_len = context_len.checked_add(query_len).ok_or_else(|| {
            execution_error(format!(
                "prefill context length {context_len} plus query length {query_len} overflows u32"
            ))
        })?;
        let state_slot = StateSlot::new(0);
        let positions = (context_len..sequence_len).collect();
        let kv_write_slots = vec![None; token_count];
        let logits = prefill_logits(action.logits, token_count, top_k);
        let execution_sequences = vec![ExecutionSequence::new(
            state_slot,
            ForwardPhase::Prefill,
            0..query_len,
            context_len,
            sequence_len,
            0..0,
        )];
        let scheduled_sequences = vec![ScheduledSequence {
            state_slot,
            request_id: action.request_id,
            session_id: action.session_id,
            kv_handle: action.kv_handle,
        }];
        Self::validate_correlation_parts(&execution_sequences, &scheduled_sequences, token_count)?;

        let tokens = std::mem::take(&mut action.tokens);
        Ok(Self {
            execution: ExecutionBatch::new(
                ForwardMode::Prefill,
                tokens,
                positions,
                kv_write_slots,
                logits,
                execution_sequences,
                Vec::new(),
            ),
            sequences: scheduled_sequences,
        })
    }

    fn from_decode(actions: &[super::actions::DecodeAction], top_k: NonZeroU32) -> Result<Self> {
        checked_u32(actions.len(), "decode row count")?;

        let mut token_ids = Vec::with_capacity(actions.len());
        let mut positions = Vec::with_capacity(actions.len());
        let mut logits = Vec::with_capacity(actions.len());
        let mut execution_sequences = Vec::with_capacity(actions.len());
        let mut scheduled_sequences = Vec::with_capacity(actions.len());

        for (index, action) in actions.iter().enumerate() {
            let input_row = checked_u32(index, "decode input row")?;
            let query_end = input_row.checked_add(1).ok_or_else(|| {
                execution_error(format!(
                    "decode input row {input_row} overflows its query range"
                ))
            })?;
            let context_len = checked_u32(action.position, "decode position")?;
            let sequence_len = context_len.checked_add(1).ok_or_else(|| {
                execution_error(format!(
                    "decode context length {context_len} plus one token overflows u32"
                ))
            })?;
            let state_slot = StateSlot::new(input_row);

            token_ids.push(action.token_id);
            positions.push(context_len);
            logits.push(if action.require_logits {
                LogitsRequest::TopK(top_k)
            } else {
                LogitsRequest::None
            });
            execution_sequences.push(ExecutionSequence::new(
                state_slot,
                ForwardPhase::Decode,
                input_row..query_end,
                context_len,
                sequence_len,
                0..0,
            ));
            scheduled_sequences.push(ScheduledSequence {
                state_slot,
                request_id: action.request_id,
                session_id: action.session_id,
                kv_handle: action.kv_handle,
            });
        }

        Ok(Self {
            execution: ExecutionBatch::new(
                ForwardMode::Decode,
                token_ids,
                positions,
                vec![None; actions.len()],
                logits,
                execution_sequences,
                Vec::new(),
            ),
            sequences: scheduled_sequences,
        })
    }

    fn validate_correlation(&self) -> Result<()> {
        Self::validate_correlation_parts(
            self.execution.sequences(),
            &self.sequences,
            self.execution.len(),
        )
    }

    fn validate_correlation_parts(
        execution_sequences: &[ExecutionSequence],
        scheduled_sequences: &[ScheduledSequence],
        row_count: usize,
    ) -> Result<()> {
        if execution_sequences.len() != scheduled_sequences.len() {
            return Err(execution_error(format!(
                "scheduled-sequence correlation has {} entries for {} execution sequences",
                scheduled_sequences.len(),
                execution_sequences.len()
            )));
        }

        for (index, (execution_sequence, scheduled_sequence)) in execution_sequences
            .iter()
            .zip(scheduled_sequences)
            .enumerate()
        {
            let expected_slot = StateSlot::new(checked_u32(index, "correlation state slot")?);
            if execution_sequence.state_slot != expected_slot {
                return Err(execution_error(format!(
                    "execution sequence {index} uses state slot {}, expected dense slot {}",
                    execution_sequence.state_slot.get(),
                    expected_slot.get()
                )));
            }
            if scheduled_sequence.state_slot != execution_sequence.state_slot {
                return Err(execution_error(format!(
                    "scheduled sequence {index} uses state slot {}, but execution uses {}",
                    scheduled_sequence.state_slot.get(),
                    execution_sequence.state_slot.get()
                )));
            }
        }

        let row_count = checked_u32(row_count, "packed input row count")?;
        for input_row in 0..row_count {
            if Self::sequence_for_input_row_parts(
                execution_sequences,
                scheduled_sequences,
                input_row,
            )
            .is_none()
            {
                return Err(execution_error(format!(
                    "packed input row {input_row} has no unique scheduled-sequence correlation"
                )));
            }
        }

        Ok(())
    }
}

fn prefill_logits(
    selection: LogitsSelection,
    token_count: usize,
    top_k: NonZeroU32,
) -> Vec<LogitsRequest> {
    match selection {
        LogitsSelection::None => vec![LogitsRequest::None; token_count],
        LogitsSelection::Last => {
            let mut requests = vec![LogitsRequest::None; token_count];
            requests[token_count - 1] = LogitsRequest::TopK(top_k);
            requests
        }
        LogitsSelection::All => vec![LogitsRequest::TopK(top_k); token_count],
    }
}

fn checked_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        execution_error(format!(
            "{label} {value} cannot be represented by the neutral u32 ABI"
        ))
    })
}

fn execution_error(message: impl Into<String>) -> Error {
    Error::Execution(message.into())
}

#[cfg(test)]
mod tests {
    use ferrule_common::execution::{LogitsOutput, LogitsRow};

    use super::*;
    use crate::scheduling::actions::{DecodeAction, PrefillChunkAction};
    use crate::scheduling::session::SequenceFinishReason;

    fn top_k() -> NonZeroU32 {
        NonZeroU32::new(4).unwrap()
    }

    fn prefill_action(logits: LogitsSelection) -> SchedulerAction {
        SchedulerAction::PrefillChunk(PrefillChunkAction {
            request_id: Some(RequestId(11)),
            session_id: SessionId(22),
            token_range: 0..3,
            position_start: 4,
            tokens: vec![10, 11, 12],
            kv_handle: Some(KvHandle(7)),
            logits,
        })
    }

    fn decode_action(
        request_id: u64,
        session_id: u64,
        token_id: u32,
        position: usize,
        kv_handle: usize,
        require_logits: bool,
    ) -> DecodeAction {
        DecodeAction {
            request_id: Some(RequestId(request_id)),
            session_id: SessionId(session_id),
            token_id,
            logit: None,
            position,
            kv_handle: Some(KvHandle(kv_handle)),
            require_logits,
        }
    }

    fn assert_execution_error<T>(result: Result<T>) {
        assert!(matches!(result, Err(Error::Execution(_))));
    }

    #[test]
    fn lowers_prefill_without_logits() {
        let mut action = prefill_action(LogitsSelection::None);
        let batch = ScheduledBatch::from_action(&mut action, top_k())
            .unwrap()
            .unwrap();

        let SchedulerAction::PrefillChunk(prefill) = &action else {
            unreachable!();
        };
        assert!(prefill.tokens.is_empty());
        assert_eq!(batch.execution().mode(), ForwardMode::Prefill);
        assert_eq!(batch.execution().token_ids(), &[10, 11, 12]);
        assert_eq!(batch.execution().positions(), &[4, 5, 6]);
        assert_eq!(batch.execution().kv_write_slots(), &[None, None, None]);
        assert_eq!(
            batch.execution().logits(),
            &[
                LogitsRequest::None,
                LogitsRequest::None,
                LogitsRequest::None
            ]
        );
        assert!(batch.execution().kv_block_ids().is_empty());
        assert_eq!(
            batch.execution().sequences(),
            &[ExecutionSequence::new(
                StateSlot::new(0),
                ForwardPhase::Prefill,
                0..3,
                4,
                7,
                0..0,
            )]
        );
        assert_eq!(
            batch.sequences(),
            &[ScheduledSequence {
                state_slot: StateSlot::new(0),
                request_id: Some(RequestId(11)),
                session_id: SessionId(22),
                kv_handle: Some(KvHandle(7)),
            }]
        );
    }

    #[test]
    fn lowers_prefill_with_last_logits() {
        let mut action = prefill_action(LogitsSelection::Last);
        let batch = ScheduledBatch::from_action(&mut action, top_k())
            .unwrap()
            .unwrap();

        assert_eq!(
            batch.execution().logits(),
            &[
                LogitsRequest::None,
                LogitsRequest::None,
                LogitsRequest::TopK(top_k()),
            ]
        );
    }

    #[test]
    fn lowers_prefill_with_all_logits() {
        let mut action = prefill_action(LogitsSelection::All);
        let batch = ScheduledBatch::from_action(&mut action, top_k())
            .unwrap()
            .unwrap();

        assert_eq!(
            batch.execution().logits(),
            &[
                LogitsRequest::TopK(top_k()),
                LogitsRequest::TopK(top_k()),
                LogitsRequest::TopK(top_k()),
            ]
        );
    }

    #[test]
    fn lowers_multi_row_decode_in_action_order() {
        let actions = vec![
            decode_action(1, 10, 41, 3, 5, true),
            decode_action(2, 20, 42, 9, 6, false),
            decode_action(3, 30, 43, 12, 7, true),
        ];
        let mut action = SchedulerAction::DecodeBatch(actions);
        let batch = ScheduledBatch::from_action(&mut action, top_k())
            .unwrap()
            .unwrap();

        assert_eq!(batch.execution().mode(), ForwardMode::Decode);
        assert_eq!(batch.execution().token_ids(), &[41, 42, 43]);
        assert_eq!(batch.execution().positions(), &[3, 9, 12]);
        assert_eq!(batch.execution().kv_write_slots(), &[None, None, None]);
        assert_eq!(
            batch.execution().logits(),
            &[
                LogitsRequest::TopK(top_k()),
                LogitsRequest::None,
                LogitsRequest::TopK(top_k()),
            ]
        );
        assert_eq!(
            batch.execution().sequences(),
            &[
                ExecutionSequence::new(StateSlot::new(0), ForwardPhase::Decode, 0..1, 3, 4, 0..0,),
                ExecutionSequence::new(StateSlot::new(1), ForwardPhase::Decode, 1..2, 9, 10, 0..0,),
                ExecutionSequence::new(StateSlot::new(2), ForwardPhase::Decode, 2..3, 12, 13, 0..0,),
            ]
        );
        assert_eq!(
            batch
                .sequences()
                .iter()
                .map(|sequence| sequence.session_id)
                .collect::<Vec<_>>(),
            vec![SessionId(10), SessionId(20), SessionId(30)]
        );
        assert_eq!(
            batch
                .sequences()
                .iter()
                .map(|sequence| sequence.kv_handle)
                .collect::<Vec<_>>(),
            vec![Some(KvHandle(5)), Some(KvHandle(6)), Some(KvHandle(7))]
        );
        assert!(batch.execution().kv_block_ids().is_empty());
    }

    #[test]
    fn terminal_and_empty_decode_actions_do_not_lower_or_mutate_actions() {
        let mut finish = SchedulerAction::Finish {
            request_id: Some(RequestId(1)),
            session_id: SessionId(1),
            reason: SequenceFinishReason::Eos,
        };
        let expected_finish = finish.clone();
        assert!(ScheduledBatch::from_action(&mut finish, top_k())
            .unwrap()
            .is_none());
        assert_eq!(finish, expected_finish);

        let mut cancel = SchedulerAction::Cancel {
            request_id: Some(RequestId(2)),
            session_id: SessionId(2),
        };
        let expected_cancel = cancel.clone();
        assert!(ScheduledBatch::from_action(&mut cancel, top_k())
            .unwrap()
            .is_none());
        assert_eq!(cancel, expected_cancel);

        let mut empty_decode = SchedulerAction::DecodeBatch(Vec::new());
        assert!(ScheduledBatch::from_action(&mut empty_decode, top_k())
            .unwrap()
            .is_none());
        assert_eq!(empty_decode, SchedulerAction::DecodeBatch(Vec::new()));
    }

    #[test]
    fn rejects_invalid_prefill_and_overflow_without_taking_payloads() {
        let mut empty = prefill_action(LogitsSelection::None);
        let SchedulerAction::PrefillChunk(action) = &mut empty else {
            unreachable!();
        };
        action.tokens.clear();
        action.token_range = 0..0;
        assert_execution_error(ScheduledBatch::from_action(&mut empty, top_k()));

        let mut mismatched_range = prefill_action(LogitsSelection::None);
        let SchedulerAction::PrefillChunk(action) = &mut mismatched_range else {
            unreachable!();
        };
        action.token_range = 0..2;
        assert_execution_error(ScheduledBatch::from_action(&mut mismatched_range, top_k()));
        let SchedulerAction::PrefillChunk(action) = &mismatched_range else {
            unreachable!();
        };
        assert_eq!(action.tokens, vec![10, 11, 12]);

        let mut prefill_overflow = prefill_action(LogitsSelection::None);
        let SchedulerAction::PrefillChunk(action) = &mut prefill_overflow else {
            unreachable!();
        };
        action.position_start = u32::MAX as usize;
        action.tokens = vec![1];
        action.token_range = 0..1;
        assert_execution_error(ScheduledBatch::from_action(&mut prefill_overflow, top_k()));
        let SchedulerAction::PrefillChunk(action) = &prefill_overflow else {
            unreachable!();
        };
        assert_eq!(action.tokens, vec![1]);

        let mut decode_overflow =
            SchedulerAction::DecodeBatch(vec![decode_action(1, 1, 1, u32::MAX as usize, 0, true)]);
        let expected_decode = decode_overflow.clone();
        assert_execution_error(ScheduledBatch::from_action(&mut decode_overflow, top_k()));
        assert_eq!(decode_overflow, expected_decode);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn rejects_positions_that_do_not_fit_the_neutral_abi_without_taking_payloads() {
        let mut action = prefill_action(LogitsSelection::None);
        let SchedulerAction::PrefillChunk(prefill) = &mut action else {
            unreachable!();
        };
        prefill.position_start = u32::MAX as usize + 1;
        assert_execution_error(ScheduledBatch::from_action(&mut action, top_k()));
        let SchedulerAction::PrefillChunk(prefill) = &action else {
            unreachable!();
        };
        assert_eq!(prefill.tokens, vec![10, 11, 12]);

        let mut action = SchedulerAction::DecodeBatch(vec![decode_action(
            1,
            1,
            1,
            u32::MAX as usize + 1,
            0,
            true,
        )]);
        let expected = action.clone();
        assert_execution_error(ScheduledBatch::from_action(&mut action, top_k()));
        assert_eq!(action, expected);
    }

    #[test]
    fn correlates_state_slots_and_packed_input_rows() {
        let mut prefill_action = prefill_action(LogitsSelection::All);
        let prefill = ScheduledBatch::from_action(&mut prefill_action, top_k())
            .unwrap()
            .unwrap();
        for input_row in 0..3 {
            assert_eq!(
                prefill.sequence_for_input_row(input_row),
                prefill.sequence_for_state_slot(StateSlot::new(0))
            );
        }
        assert!(prefill.sequence_for_input_row(3).is_none());
        assert!(prefill.sequence_for_state_slot(StateSlot::new(1)).is_none());

        let mut decode_action = SchedulerAction::DecodeBatch(vec![
            decode_action(1, 10, 41, 3, 5, true),
            decode_action(2, 20, 42, 9, 6, true),
        ]);
        let decode = ScheduledBatch::from_action(&mut decode_action, top_k())
            .unwrap()
            .unwrap();
        assert_eq!(
            decode.sequence_for_input_row(0).unwrap().request_id,
            Some(RequestId(1))
        );
        assert_eq!(
            decode.sequence_for_input_row(1).unwrap().request_id,
            Some(RequestId(2))
        );
        assert!(decode.sequence_for_input_row(2).is_none());
    }

    #[test]
    fn validate_output_checks_common_contract_before_correlation() {
        let mut action = prefill_action(LogitsSelection::Last);
        let batch = ScheduledBatch::from_action(&mut action, top_k())
            .unwrap()
            .unwrap();
        let invalid = ExecutionOutput::new(vec![LogitsRow::new(2, LogitsOutput::Full(Vec::new()))]);
        assert_execution_error(batch.validate_output(&invalid));

        let valid = ExecutionOutput::new(vec![LogitsRow::new(2, LogitsOutput::TopK(Vec::new()))]);
        batch.validate_output(&valid).unwrap();

        let mut missing_correlation = batch.clone();
        missing_correlation.sequences.clear();
        let error = missing_correlation.validate_output(&invalid).unwrap_err();
        assert!(matches!(error, Error::Execution(message) if message.contains("requested top-k")));
        let error = missing_correlation.validate_output(&valid).unwrap_err();
        assert!(matches!(error, Error::Execution(message) if message.contains("correlation")));
    }
}
