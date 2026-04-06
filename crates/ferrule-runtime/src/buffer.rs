use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceSample {
    pub prompt_ids: Vec<u32>,
    pub completion_ids: Vec<u32>,
    pub completion_text: String,
    pub token_logprobs: Vec<f32>,
    pub reward: f32,
    pub finish_reason: String,
}

impl SequenceSample {
    pub fn completion_len(&self) -> usize {
        self.completion_ids.len()
    }

    pub fn logprob_sum(&self) -> f32 {
        self.token_logprobs.iter().copied().sum()
    }
}

#[derive(Debug, Default)]
pub struct OnPolicyBuffer {
    samples: Vec<SequenceSample>,
}

#[derive(Debug, Clone)]
pub struct BufferStats {
    pub num_samples: usize,
    pub mean_reward: f32,
    pub mean_completion_tokens: f32,
}

impl OnPolicyBuffer {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    pub fn push(&mut self, sample: SequenceSample) {
        self.samples.push(sample);
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn clear(&mut self) {
        self.samples.clear();
    }

    pub fn samples(&self) -> &[SequenceSample] {
        &self.samples
    }

    pub fn into_samples(self) -> Vec<SequenceSample> {
        self.samples
    }

    pub fn stats(&self) -> BufferStats {
        if self.samples.is_empty() {
            return BufferStats {
                num_samples: 0,
                mean_reward: 0.0,
                mean_completion_tokens: 0.0,
            };
        }

        let n = self.samples.len() as f32;
        let reward_sum = self.samples.iter().map(|s| s.reward).sum::<f32>();
        let token_sum = self
            .samples
            .iter()
            .map(|s| s.completion_len() as f32)
            .sum::<f32>();

        BufferStats {
            num_samples: self.samples.len(),
            mean_reward: reward_sum / n,
            mean_completion_tokens: token_sum / n,
        }
    }

    pub fn mean_reward(&self) -> f32 {
        self.stats().mean_reward
    }
}
