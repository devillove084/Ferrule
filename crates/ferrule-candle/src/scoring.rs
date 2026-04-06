use candle_core::Tensor;
use candle_nn::ops::log_softmax;
use ferrule_core::{FerruleError, FerruleResult};

pub fn token_logprob_from_logits(logits: &Tensor, token_id: u32) -> FerruleResult<f32> {
    let logits = if logits.rank() == 2 {
        logits
            .squeeze(0)
            .map_err(|e| FerruleError::Model(format!("failed to squeeze logits: {e}")))?
    } else {
        logits.clone()
    };

    let log_probs = log_softmax(&logits, 0)
        .map_err(|e| FerruleError::Model(format!("log_softmax failed: {e}")))?;

    let values = log_probs
        .to_vec1::<f32>()
        .map_err(|e| FerruleError::Model(format!("failed to convert log_probs to vec: {e}")))?;

    values
        .get(token_id as usize)
        .copied()
        .ok_or_else(|| FerruleError::Model(format!("token id out of range: {token_id}")))
}
