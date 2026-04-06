pub fn sequence_reinforce_loss(logprobs: &[f32], reward: f32, baseline: f32) -> f32 {
    let advantage = reward - baseline;
    let logprob_sum = logprobs.iter().copied().sum::<f32>();
    -advantage * logprob_sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reinforce_loss_sign_matches_advantage() {
        let logprobs = vec![-0.2, -0.4];
        let loss_pos_adv = sequence_reinforce_loss(&logprobs, 1.0, 0.0);
        let loss_neg_adv = sequence_reinforce_loss(&logprobs, 0.0, 1.0);

        assert!(loss_pos_adv > 0.0);
        assert!(loss_neg_adv < 0.0);
    }
}
