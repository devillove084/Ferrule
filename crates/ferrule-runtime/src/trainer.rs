use crate::buffer::SequenceSample;
use ferrule_core::{FerruleError, FerruleResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaselineMode {
    Zero,
    BatchMean,
    LeaveOneOut,
}

impl BaselineMode {
    pub fn parse(s: &str) -> FerruleResult<Self> {
        match s {
            "zero" => Ok(Self::Zero),
            "batch_mean" => Ok(Self::BatchMean),
            "leave_one_out" => Ok(Self::LeaveOneOut),
            other => Err(FerruleError::Config(format!(
                "unsupported baseline_mode: {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SampleTrainStats {
    pub reward: f32,
    pub baseline: f32,
    pub advantage: f32,
    pub logprob_sum: f32,
    pub objective: f32,
}

#[derive(Debug, Clone)]
pub struct BatchTrainStats {
    pub mean_reward: f32,
    pub mean_advantage: f32,
    pub mean_logprob_sum: f32,
    pub mean_objective: f32,
}

pub fn reinforce_objective(sample: &SequenceSample, baseline: f32) -> SampleTrainStats {
    let logprob_sum = sample.logprob_sum();
    let advantage = sample.reward - baseline;
    let objective = advantage * logprob_sum;

    SampleTrainStats {
        reward: sample.reward,
        baseline,
        advantage,
        logprob_sum,
        objective,
    }
}

pub fn batch_reinforce_stats(
    samples: &[SequenceSample],
    baseline_mode: BaselineMode,
) -> BatchTrainStats {
    if samples.is_empty() {
        return BatchTrainStats {
            mean_reward: 0.0,
            mean_advantage: 0.0,
            mean_logprob_sum: 0.0,
            mean_objective: 0.0,
        };
    }

    let reward_sum = samples.iter().map(|s| s.reward).sum::<f32>();
    let n = samples.len() as f32;

    let per_sample = samples
        .iter()
        .enumerate()
        .map(|(idx, s)| {
            let baseline = match baseline_mode {
                BaselineMode::Zero => 0.0,
                BaselineMode::BatchMean => reward_sum / n,
                BaselineMode::LeaveOneOut => {
                    if samples.len() == 1 {
                        0.0
                    } else {
                        (reward_sum - s.reward) / ((samples.len() - 1) as f32)
                    }
                }
            };
            let _ = idx;
            reinforce_objective(s, baseline)
        })
        .collect::<Vec<_>>();

    BatchTrainStats {
        mean_reward: per_sample.iter().map(|s| s.reward).sum::<f32>() / n,
        mean_advantage: per_sample.iter().map(|s| s.advantage).sum::<f32>() / n,
        mean_logprob_sum: per_sample.iter().map(|s| s.logprob_sum).sum::<f32>() / n,
        mean_objective: per_sample.iter().map(|s| s.objective).sum::<f32>() / n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::SequenceSample;

    #[test]
    fn zero_baseline_keeps_single_sample_advantage_nonzero() {
        let sample = SequenceSample {
            prompt_ids: vec![1],
            completion_ids: vec![2, 3],
            completion_text: "ok".to_string(),
            token_logprobs: vec![-0.2, -0.3],
            reward: 1.0,
            finish_reason: "done".to_string(),
        };

        let stats = batch_reinforce_stats(&[sample], BaselineMode::Zero);
        assert!(stats.mean_advantage > 0.0);
    }

    #[test]
    fn batch_mean_baseline_zeroes_single_sample_advantage() {
        let sample = SequenceSample {
            prompt_ids: vec![1],
            completion_ids: vec![2],
            completion_text: "ok".to_string(),
            token_logprobs: vec![-0.2],
            reward: 1.0,
            finish_reason: "done".to_string(),
        };

        let stats = batch_reinforce_stats(&[sample], BaselineMode::BatchMean);
        assert_eq!(stats.mean_advantage, 0.0);
    }
}
