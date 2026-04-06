use ferrule_core::{Event, FerruleResult, RewardFn, RewardOutput, Trajectory, async_trait};

pub struct FinishReward;

#[async_trait]
impl RewardFn<Trajectory> for FinishReward {
    fn name(&self) -> &'static str {
        "finish_reward_v1"
    }

    async fn evaluate(&self, traj: &Trajectory) -> FerruleResult<RewardOutput> {
        let has_finish = traj
            .events
            .iter()
            .any(|e| matches!(e, Event::Finish { .. }));
        let reward = if has_finish { 1.0 } else { 0.0 };

        Ok(RewardOutput {
            total_reward: reward,
            components: vec![("finished".to_string(), reward)],
            version: "v1".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrule_core::{EpisodeId, Event, Trajectory};

    #[tokio::test]
    async fn finish_reward_gives_one_when_finish_exists() {
        let mut traj = Trajectory::new(EpisodeId::new());
        traj.events.push(Event::Finish {
            reason: "done".to_string(),
        });

        let reward = FinishReward.evaluate(&traj).await.unwrap();
        assert_eq!(reward.total_reward, 1.0);
    }
}
