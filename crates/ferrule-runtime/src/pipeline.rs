use crate::{OnPolicyBuffer, SequenceSample};
use ferrule_core::{FerruleError, FerruleResult};

pub fn make_sequence_sample(
    prompt_ids: Vec<u32>,
    completion_ids: Vec<u32>,
    completion_text: String,
    token_logprobs: Vec<f32>,
    reward: f32,
    finish_reason: String,
) -> FerruleResult<SequenceSample> {
    if completion_ids.len() != token_logprobs.len() {
        return Err(FerruleError::Runtime(format!(
            "completion_ids len {} != token_logprobs len {}",
            completion_ids.len(),
            token_logprobs.len()
        )));
    }

    Ok(SequenceSample {
        prompt_ids,
        completion_ids,
        completion_text,
        token_logprobs,
        reward,
        finish_reason,
    })
}

pub fn push_sequence_sample(buffer: &mut OnPolicyBuffer, sample: SequenceSample) {
    buffer.push(sample);
}
