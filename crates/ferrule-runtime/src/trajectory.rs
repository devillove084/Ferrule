use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStepRecord {
    pub step_idx: usize,
    pub prompt_text: String,
    pub action_text: String,
    pub action_token_ids: Vec<u32>,
    pub reward_delta: f32,
    pub cumulative_reward: f32,
    pub done: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTrajectory {
    pub initial_observation: String,
    pub steps: Vec<AgentStepRecord>,
    pub total_reward: f32,
    pub finished: bool,
}
