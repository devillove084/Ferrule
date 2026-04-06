use crate::FerruleResult;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogFormat {
    Pretty,
    Json,
}

impl Default for LogFormat {
    fn default() -> Self {
        Self::Json
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    pub service_name: String,
    #[serde(default = "default_log_level")]
    pub log_level: String,
    #[serde(default)]
    pub log_format: LogFormat,
    #[serde(default = "default_metrics_enabled")]
    pub metrics_enabled: bool,
    #[serde(default = "default_metrics_bind")]
    pub metrics_bind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub backend: String,
    pub model_id: String,

    #[serde(default = "default_model_family")]
    pub family: String,

    #[serde(default = "default_device")]
    pub device: String,

    #[serde(default)]
    pub revision: Option<String>,

    #[serde(default)]
    pub tokenizer_path: Option<String>,

    #[serde(default)]
    pub weights_path: Option<String>,

    #[serde(default)]
    pub config_path: Option<String>,

    #[serde(default = "default_chat_template")]
    pub chat_template: String,

    #[serde(default = "default_dtype")]
    pub dtype: String,

    #[serde(default)]
    pub use_flash_attn: bool,

    #[serde(default = "default_use_kv_cache")]
    pub use_kv_cache: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutConfig {
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub expected_substring: Option<String>,
    #[serde(default)]
    pub dataset_path: Option<String>,
    #[serde(default = "default_num_samples")]
    pub num_samples: usize,
    #[serde(default = "default_train_max_new_tokens")]
    pub max_new_tokens: usize,
    #[serde(default = "default_baseline_mode")]
    pub baseline_mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    #[serde(default = "default_agent_initial_observation")]
    pub initial_observation: String,
    #[serde(default = "default_agent_max_steps")]
    pub max_steps: usize,
    #[serde(default = "default_agent_max_new_tokens")]
    pub max_new_tokens: usize,
    #[serde(default = "default_agent_temperature")]
    pub temperature: f32,
    #[serde(default = "default_agent_top_p")]
    pub top_p: Option<f32>,
    #[serde(default = "default_agent_top_k")]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub expected_final: Option<String>,

    #[serde(default)]
    pub dataset_path: Option<String>,

    #[serde(default = "default_agent_num_episodes")]
    pub num_episodes: usize,

    #[serde(default = "default_agent_gamma")]
    pub gamma: f32,

    #[serde(default = "default_agent_baseline_mode")]
    pub baseline_mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub observability: ObservabilityConfig,
    pub model: ModelConfig,
    pub rollout: RolloutConfig,
    #[serde(default)]
    pub train: Option<TrainConfig>,
    #[serde(default)]
    pub agent: Option<AgentConfig>,
}

impl AppConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> FerruleResult<Self> {
        let raw = fs::read_to_string(path)?;
        let cfg = toml::from_str::<Self>(&raw)?;
        Ok(cfg)
    }
}

fn default_agent_initial_observation() -> String {
    "Use tools if needed, then provide FINAL:<answer>.".to_string()
}

fn default_agent_max_steps() -> usize {
    4
}

fn default_agent_max_new_tokens() -> usize {
    32
}

fn default_agent_temperature() -> f32 {
    0.2
}

fn default_agent_top_p() -> Option<f32> {
    Some(0.9)
}

fn default_agent_top_k() -> Option<usize> {
    Some(20)
}

fn default_agent_num_episodes() -> usize {
    8
}

fn default_agent_gamma() -> f32 {
    1.0
}

fn default_agent_baseline_mode() -> String {
    "trajectory_leave_one_out".to_string()
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_metrics_enabled() -> bool {
    true
}

fn default_metrics_bind() -> String {
    "127.0.0.1:9000".to_string()
}

fn default_device() -> String {
    "cpu".to_string()
}

fn default_model_family() -> String {
    "llama".to_string()
}

fn default_chat_template() -> String {
    "plain".to_string()
}

fn default_dtype() -> String {
    "f16".to_string()
}

fn default_use_kv_cache() -> bool {
    true
}

fn default_max_steps() -> usize {
    8
}

fn default_seed() -> u64 {
    42
}

fn default_num_samples() -> usize {
    4
}

fn default_train_max_new_tokens() -> usize {
    32
}

fn default_baseline_mode() -> String {
    "leave_one_out".to_string()
}
