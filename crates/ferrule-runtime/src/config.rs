//! Load generation defaults from model-side generation_config.json.

use crate::sampler::SamplingConfig;
use std::path::Path;

/// Generation defaults loaded from the model directory.
/// CLI flags take precedence over these values.
#[derive(Debug, Clone, Default)]
pub struct ModelGenerationDefaults {
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub max_new_tokens: Option<usize>,
}

impl ModelGenerationDefaults {
    pub fn load(model_dir: &Path) -> Option<Self> {
        let path = model_dir.join("generation_config.json");
        let text = std::fs::read_to_string(path).ok()?;
        let json: serde_json::Value = serde_json::from_str(&text).ok()?;

        let float = |key: &str| json.get(key)?.as_f64().map(|v| v as f32);
        let int = |key: &str| json.get(key)?.as_u64().map(|v| v as usize);
        let token = |key: &str| json.get(key)?.as_u64().map(|v| v as u32);

        Some(Self {
            eos_token_id: token("eos_token_id"),
            pad_token_id: token("pad_token_id"),
            temperature: float("temperature"),
            top_k: int("top_k"),
            top_p: float("top_p"),
            repetition_penalty: float("repetition_penalty"),
            max_new_tokens: int("max_new_tokens"),
        })
    }

    /// Merge model defaults into a SamplingConfig, using model values only
    /// where SamplingConfig is at its own default.
    pub fn apply_to_config(&self, config: &mut SamplingConfig) {
        let d = SamplingConfig::default();
        if let Some(t) = self.temperature {
            if (config.temperature - d.temperature).abs() < f32::EPSILON {
                config.temperature = t;
            }
        }
        if let Some(k) = self.top_k {
            if config.top_k == d.top_k {
                config.top_k = k;
            }
        }
        if let Some(p) = self.top_p {
            if (config.top_p - d.top_p).abs() < f32::EPSILON {
                config.top_p = p;
            }
        }
        if let Some(rp) = self.repetition_penalty {
            if (config.repeat_penalty - d.repeat_penalty).abs() < f32::EPSILON {
                config.repeat_penalty = rp;
            }
        }
    }
}
