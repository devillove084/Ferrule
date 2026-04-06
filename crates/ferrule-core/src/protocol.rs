use crate::EpisodeId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub seed: u64,
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub max_new_tokens: usize,
    pub stop_strings: Vec<String>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            seed: 42,
            temperature: 0.8,
            top_p: Some(0.95),
            top_k: None,
            max_new_tokens: 256,
            stop_strings: vec![],
            repeat_penalty: 1.0,
            repeat_last_n: 64,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelOutput {
    Text {
        content: String,
    },
    CallTool {
        name: String,
        arguments_json: String,
    },
    Finish {
        reason: String,
        final_text: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStep {
    pub action: ModelOutput,
    pub usage: TokenUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentAction {
    CallTool {
        name: String,
        arguments_json: String,
    },
    Finish {
        final_text: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub text: String,
    pub done: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub observation: Observation,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    pub info_json: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StopReason {
    ModelFinish,
    EnvTerminated,
    MaxSteps,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Event {
    Reset {
        observation: String,
    },
    ModelText {
        content: String,
    },
    ToolCall {
        name: String,
        arguments_json: String,
    },
    ToolResult {
        content: String,
    },
    Finish {
        reason: String,
    },
    Error {
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardOutput {
    pub total_reward: f32,
    pub components: Vec<(String, f32)>,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    pub episode_id: EpisodeId,
    pub events: Vec<Event>,
    pub total_reward: f32,
    pub stop_reason: Option<StopReason>,
    pub reward: Option<RewardOutput>,
    pub done: bool,
}

impl Trajectory {
    pub fn new(episode_id: EpisodeId) -> Self {
        Self {
            episode_id,
            events: Vec::new(),
            total_reward: 0.0,
            stop_reason: None,
            reward: None,
            done: false,
        }
    }
}
