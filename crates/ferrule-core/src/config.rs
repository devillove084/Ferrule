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

fn default_dtype() -> String {
    "f16".to_string()
}

fn default_use_kv_cache() -> bool {
    true
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
pub struct AppConfig {
    pub observability: ObservabilityConfig,
    pub model: ModelConfig,
    pub rollout: RolloutConfig,
}

impl AppConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> FerruleResult<Self> {
        let raw = fs::read_to_string(path)?;
        let cfg = toml::from_str::<Self>(&raw)?;
        Ok(cfg)
    }
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

fn default_max_steps() -> usize {
    8
}

fn default_seed() -> u64 {
    42
}
