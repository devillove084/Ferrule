#![allow(clippy::needless_range_loop)]
//! OLMoE — f32 weights, tokenizer.
pub mod config;
pub mod cpu_forward;
pub mod loader;
pub mod weights;

// Re-exports — keep the same public API surface as before the split.
pub use config::OlmoeConfig;
pub use cpu_forward::rms_norm;
pub use weights::{AttnWeights, ExpertWeights, LayerWeights, LinearWeight};

use ferrule_core::Result;
use std::path::PathBuf;
use tokenizers::Tokenizer;

pub struct OlmoeModel {
    pub config: OlmoeConfig,
    pub embed: Vec<f32>,
    pub lm_head: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub model_dir: PathBuf,
    pub(crate) tokenizer: Tokenizer,
}

impl OlmoeModel {
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| ferrule_core::Error::Model(format!("encode: {e}")))
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.config.eos_token_id
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| ferrule_core::Error::Model(format!("decode: {e}")))
    }

    /// Extract the tokenizer, consuming the model.
    pub fn into_tokenizer(self) -> Tokenizer {
        self.tokenizer
    }
}
