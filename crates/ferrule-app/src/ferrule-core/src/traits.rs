use crate::{
    AgentAction, FerruleResult, ModelStep, Observation, RewardOutput, SamplingParams, StepResult,
    async_trait,
};

#[async_trait]
pub trait PolicyModel: Send + Sync {
    type Session: Send;

    fn name(&self) -> &str;

    async fn new_session(
        &self,
        prompt_token_ids: &[u32],
        params: &SamplingParams,
    ) -> FerruleResult<Self::Session>;

    async fn step(&self, session: &mut Self::Session) -> FerruleResult<ModelStep>;
}

#[async_trait]
pub trait Environment: Send {
    async fn reset(&mut self, seed: u64) -> FerruleResult<Observation>;
    async fn step(&mut self, action: AgentAction) -> FerruleResult<StepResult>;
}

#[async_trait]
pub trait RewardFn<T>: Send + Sync {
    fn name(&self) -> &'static str;
    async fn evaluate(&self, input: &T) -> FerruleResult<RewardOutput>;
}
