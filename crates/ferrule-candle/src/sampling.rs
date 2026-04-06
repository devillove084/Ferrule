use candle_core::Tensor;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use ferrule_core::{FerruleError, FerruleResult, SamplingParams};

pub fn build_logits_processor(params: &SamplingParams) -> LogitsProcessor {
    let temperature = params.temperature as f64;

    let sampling = if temperature <= 0.0 {
        Sampling::ArgMax
    } else {
        match (params.top_k, params.top_p) {
            (None, None) => Sampling::All { temperature },
            (Some(k), None) => Sampling::TopK { k, temperature },
            (None, Some(p)) => Sampling::TopP {
                p: p as f64,
                temperature,
            },
            (Some(k), Some(p)) => Sampling::TopKThenTopP {
                k,
                p: p as f64,
                temperature,
            },
        }
    };

    LogitsProcessor::from_sampling(params.seed, sampling)
}

pub fn sample_next_token(
    logits: &Tensor,
    processor: &mut LogitsProcessor,
    prior_tokens: &[u32],
    params: &SamplingParams,
) -> FerruleResult<u32> {
    let logits = if logits.dims().len() == 2 {
        logits
            .squeeze(0)
            .map_err(|e| FerruleError::Model(format!("failed to squeeze logits: {e}")))?
    } else {
        logits.clone()
    };

    let logits = if (params.repeat_penalty - 1.0).abs() < f32::EPSILON {
        logits
    } else {
        let start_at = prior_tokens.len().saturating_sub(params.repeat_last_n);
        candle_transformers::utils::apply_repeat_penalty(
            &logits,
            params.repeat_penalty,
            &prior_tokens[start_at..],
        )
        .map_err(|e| FerruleError::Model(format!("repeat penalty failed: {e}")))?
    };

    processor
        .sample(&logits)
        .map_err(|e| FerruleError::Model(format!("sampling failed: {e}")))
}
