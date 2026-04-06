use candle_core::{IndexOp, Tensor};
use candle_nn::ops::log_softmax;
use ferrule_core::{FerruleError, FerruleResult};

pub fn sequence_logprob_sum_tensor(
    logits_per_step: &[Tensor],
    target_ids: &[u32],
) -> FerruleResult<Tensor> {
    if logits_per_step.len() != target_ids.len() {
        return Err(FerruleError::Runtime(format!(
            "logits_per_step len {} != target_ids len {}",
            logits_per_step.len(),
            target_ids.len()
        )));
    }

    let mut parts = Vec::with_capacity(target_ids.len());

    for (logits, token_id) in logits_per_step.iter().zip(target_ids.iter()) {
        let logits = if logits.rank() == 2 {
            logits
                .squeeze(0)
                .map_err(|e| FerruleError::Model(format!("failed to squeeze logits: {e}")))?
        } else {
            logits.clone()
        };

        let log_probs = log_softmax(&logits, 0)
            .map_err(|e| FerruleError::Model(format!("log_softmax failed: {e}")))?;

        let picked = log_probs
            .i(*token_id as usize)
            .map_err(|e| FerruleError::Model(format!("failed to index target token: {e}")))?;
        parts.push(picked);
    }

    let stacked = Tensor::stack(&parts, 0)
        .map_err(|e| FerruleError::Model(format!("failed to stack token logprobs: {e}")))?;

    stacked
        .sum(0)
        .map_err(|e| FerruleError::Model(format!("failed to sum token logprobs: {e}")))
}
