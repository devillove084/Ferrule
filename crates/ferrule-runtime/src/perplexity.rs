//! Perplexity computation over text corpora.
//!
//! Teacher-forces each token and accumulates negative log-likelihood.

use ferrule_core::Result;

/// Perplexity result for a text corpus.
#[derive(Debug, Clone)]
pub struct PerplexityResult {
    /// Total number of tokens evaluated (including the first, which has no loss).
    pub total_tokens: usize,
    /// Tokens that contributed to the loss (total_tokens - 1).
    pub loss_tokens: usize,
    /// Sum of negative log-likelihoods.
    pub sum_nll: f64,
    /// Perplexity = exp(sum_nll / loss_tokens).
    pub perplexity: f64,
    /// Tokens per second during evaluation.
    pub tokens_per_second: f64,
    /// Wall-clock duration.
    pub duration_secs: f64,
}

/// Compute perplexity by teacher-forcing each token through the model.
///
/// `forward` is called for each token and returns logits for the next token.
/// The log-softmax of the next token's logit is accumulated.
pub fn compute_perplexity<F>(
    tokens: &[u32],
    _vocab_size: usize,
    mut forward: F,
) -> Result<PerplexityResult>
where
    F: FnMut(u32) -> Result<Vec<f32>>,
{
    let t0 = std::time::Instant::now();
    let mut sum_nll = 0.0f64;
    let mut loss_tokens = 0usize;

    for (i, &token) in tokens.iter().enumerate() {
        let logits = forward(token)?;
        // Compute log-softmax: log(softmax(logits)[next_token])
        // = logits[next] - log(sum(exp(logits)))
        if i + 1 < tokens.len() {
            let next_token = tokens[i + 1] as usize;
            if next_token < logits.len() {
                let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum_exp = 0.0f64;
                for &l in logits.iter() {
                    sum_exp += ((l - max_logit) as f64).exp();
                }
                let log_prob = logits[next_token] as f64 - max_logit as f64 - sum_exp.ln();
                sum_nll -= log_prob;
                loss_tokens += 1;
            }
        }
    }

    let dur = t0.elapsed();
    let perplexity = if loss_tokens > 0 {
        (sum_nll / loss_tokens as f64).exp()
    } else {
        f64::INFINITY
    };

    Ok(PerplexityResult {
        total_tokens: tokens.len(),
        loss_tokens,
        sum_nll,
        perplexity,
        tokens_per_second: tokens.len() as f64 / dur.as_secs_f64().max(1e-6),
        duration_secs: dur.as_secs_f64(),
    })
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock forward: always returns uniform logits (all 0.0).
    /// For vocab_size V, uniform means perplexity = V.
    fn uniform_forward(vocab: usize) -> impl FnMut(u32) -> Result<Vec<f32>> {
        move |_token: u32| Ok(vec![0.0f32; vocab])
    }

    #[test]
    fn perplexity_uniform_distribution() {
        let vocab = 100;
        let tokens: Vec<u32> = vec![0, 1, 2, 3, 4];
        let result = compute_perplexity(&tokens, vocab, uniform_forward(vocab)).unwrap();
        assert_eq!(result.total_tokens, 5);
        assert_eq!(result.loss_tokens, 4);
        // For uniform logits [0,0,...], perplexity should equal vocab size
        assert!((result.perplexity - vocab as f64).abs() < 1.0);
    }

    #[test]
    fn perplexity_single_token_no_loss() {
        let result = compute_perplexity(&[42], 10, |_| Ok(vec![0.0f32; 10])).unwrap();
        assert_eq!(result.total_tokens, 1);
        assert_eq!(result.loss_tokens, 0);
        assert!(result.perplexity.is_infinite());
    }

    #[test]
    fn perplexity_peaked_distribution() {
        // Give high logit to the correct next token → low perplexity
        let tokens: Vec<u32> = vec![0, 1, 2];
        let mut step = 0;
        let result = compute_perplexity(&tokens.clone(), 10, move |_| {
            step += 1;
            let mut logits = vec![-100.0f32; 10];
            // Make the correct next token have high logit
            if step < tokens.len() {
                logits[tokens[step] as usize] = 0.0;
            }
            Ok(logits)
        })
        .unwrap();
        // Perplexity should be close to 1 since we're very confident
        assert!(result.perplexity < 3.0);
    }

    #[test]
    fn nll_is_positive() {
        let tokens: Vec<u32> = vec![0, 1, 2, 3];
        let result = compute_perplexity(&tokens, 50, uniform_forward(50)).unwrap();
        assert!(result.sum_nll > 0.0);
    }
}
