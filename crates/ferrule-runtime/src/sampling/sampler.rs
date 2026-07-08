use std::cmp::Ordering;

/// Per-token top-K logprobs for debugging and analysis.
#[derive(Debug, Clone, Default)]
pub struct Logprobs {
    pub token: u32,
    pub text: String,
    pub entries: Vec<(u32, f32)>,
}

#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// `0.0` means deterministic greedy sampling.
    pub temperature: f32,
    /// `0` disables top-k filtering.
    pub top_k: usize,
    /// `>= 1.0` disables nucleus filtering.
    pub top_p: f32,
    /// `0.0` disables min-p filtering.
    pub min_p: f32,
    /// `1.0` disables repeat penalty.
    pub repeat_penalty: f32,
    /// Number of most recent tokens considered by repeat penalty.
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: 0x46_65_72_72_75_6c_65,
        }
    }
}

impl SamplingConfig {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sampler {
    config: SamplingConfig,
    rng: TinyRng,
}

impl Sampler {
    pub fn new(config: SamplingConfig) -> Self {
        let rng = TinyRng::new(config.seed);
        Self { config, rng }
    }

    pub fn config(&self) -> &SamplingConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut SamplingConfig {
        &mut self.config
    }

    pub fn top_logprobs(&self, logits: &[f32], k: usize) -> Vec<(u32, f32)> {
        if logits.is_empty() || k == 0 {
            return Vec::new();
        }
        let mut pairs: Vec<(u32, f32)> = logits
            .iter()
            .copied()
            .enumerate()
            .map(|(id, logit)| (id as u32, sanitize_logit(logit)))
            .collect();
        pairs.sort_by(rank_candidates_desc);
        pairs.truncate(k);
        // Numerically stable softmax for display
        if let Some(&(_, max_logit)) = pairs.first() {
            let mut sum = 0.0f32;
            for (_, logit) in &mut pairs {
                *logit = (*logit - max_logit).exp();
                if logit.is_finite() && *logit > 0.0 {
                    sum += *logit;
                }
            }
            if sum > 0.0 {
                for (_, prob) in &mut pairs {
                    *prob /= sum;
                }
            }
        }
        pairs
    }

    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> u32 {
        if logits.is_empty() {
            return 0;
        }

        let mut candidates: Vec<(u32, f32)> = logits
            .iter()
            .copied()
            .enumerate()
            .map(|(id, logit)| (id as u32, sanitize_logit(logit)))
            .collect();

        apply_repeat_penalty(&mut candidates, history, &self.config);

        if self.config.temperature <= 0.0 {
            return greedy(&candidates);
        }

        let temperature = self.config.temperature.max(1e-6);
        for (_, logit) in &mut candidates {
            *logit /= temperature;
        }

        candidates.sort_by(rank_candidates_desc);

        if self.config.top_k > 0 && self.config.top_k < candidates.len() {
            candidates.truncate(self.config.top_k);
        }

        let mut probs = softmax_candidates(&candidates);
        apply_min_p(&mut probs, self.config.min_p);
        apply_top_p(&mut probs, self.config.top_p);

        sample_probabilities(&mut self.rng, &probs).unwrap_or_else(|| greedy(&candidates))
    }
}

fn sanitize_logit(logit: f32) -> f32 {
    if logit.is_finite() {
        logit
    } else {
        f32::NEG_INFINITY
    }
}

fn greedy(candidates: &[(u32, f32)]) -> u32 {
    candidates
        .iter()
        .copied()
        .min_by(rank_candidates_desc)
        .map(|(id, _)| id)
        .unwrap_or(0)
}

fn rank_candidates_desc(left: &(u32, f32), right: &(u32, f32)) -> Ordering {
    right
        .1
        .total_cmp(&left.1)
        .then_with(|| left.0.cmp(&right.0))
}

fn apply_repeat_penalty(candidates: &mut [(u32, f32)], history: &[u32], config: &SamplingConfig) {
    if config.repeat_penalty <= 0.0 || (config.repeat_penalty - 1.0).abs() < f32::EPSILON {
        return;
    }
    if config.repeat_last_n == 0 || history.is_empty() {
        return;
    }

    let start = history.len().saturating_sub(config.repeat_last_n);
    for &token in &history[start..] {
        let Some((_, logit)) = candidates.get_mut(token as usize) else {
            continue;
        };
        if *logit < 0.0 {
            *logit *= config.repeat_penalty;
        } else {
            *logit /= config.repeat_penalty;
        }
    }
}

fn softmax_candidates(candidates: &[(u32, f32)]) -> Vec<(u32, f32)> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let max_logit = candidates[0].1;
    let mut sum = 0.0f32;
    let mut probs = Vec::with_capacity(candidates.len());
    for &(id, logit) in candidates {
        let p = (logit - max_logit).exp();
        if p.is_finite() && p > 0.0 {
            sum += p;
            probs.push((id, p));
        }
    }

    if sum <= 0.0 {
        return candidates.iter().map(|&(id, _)| (id, 1.0)).collect();
    }

    for (_, p) in &mut probs {
        *p /= sum;
    }
    probs
}

fn apply_min_p(probs: &mut Vec<(u32, f32)>, min_p: f32) {
    if min_p <= 0.0 || probs.len() <= 1 {
        return;
    }
    let best = probs.first().copied();
    let max_prob = probs.iter().map(|(_, prob)| *prob).fold(0.0f32, f32::max);
    let threshold = max_prob * min_p;
    probs.retain(|(_, prob)| *prob >= threshold);
    if probs.is_empty() {
        if let Some(best) = best {
            probs.push(best);
        }
    }
}

fn apply_top_p(probs: &mut Vec<(u32, f32)>, top_p: f32) {
    if top_p <= 0.0 || top_p >= 1.0 || probs.len() <= 1 {
        return;
    }

    let mut cumulative = 0.0f32;
    let mut keep = 0usize;
    for (_, prob) in probs.iter() {
        cumulative += *prob;
        keep += 1;
        if cumulative >= top_p {
            break;
        }
    }
    probs.truncate(keep.max(1));
}

fn sample_probabilities(rng: &mut TinyRng, probs: &[(u32, f32)]) -> Option<u32> {
    if probs.is_empty() {
        return None;
    }

    let sum: f32 = probs.iter().map(|(_, prob)| *prob).sum();
    if sum <= 0.0 || !sum.is_finite() {
        return probs.first().map(|(id, _)| *id);
    }

    let mut threshold = rng.next_f32() * sum;
    for &(id, prob) in probs {
        threshold -= prob;
        if threshold <= 0.0 {
            return Some(id);
        }
    }
    probs.last().map(|(id, _)| *id)
}

#[derive(Debug, Clone)]
struct TinyRng {
    state: u64,
}

impl TinyRng {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn next_f32(&mut self) -> f32 {
        let value = self.next_u64() >> 40;
        value as f32 / (1u32 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_top_logprobs_share_tie_breaking() {
        let mut sampler = Sampler::new(SamplingConfig::greedy());
        let logits = vec![0.0, 2.0, 1.0, 2.0];

        assert_eq!(sampler.sample(&logits, &[]), 1);
        assert_eq!(sampler.top_logprobs(&logits, 2)[0].0, 1);
        assert_eq!(sampler.top_logprobs(&logits, 2)[1].0, 3);
    }
}
