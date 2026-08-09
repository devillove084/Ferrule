//! Thin Qwen3-MoE artifact boundary over generic decoder resources.

use std::path::Path;

use ferrule_common::{Error, Result};

use crate::transformer::{BoundDecoderResources, DecoderLoadOptions, HFDecoderCheckpoint};
use crate::{ModelFamily, TokenizerHandle};

use super::{Qwen3MoeConfig, Qwen3MoeRecipe};

pub struct Qwen3MoeCheckpoint {
    pub(super) config: Qwen3MoeConfig,
    checkpoint: HFDecoderCheckpoint,
    tokenizer: TokenizerHandle,
}

impl Qwen3MoeCheckpoint {
    pub fn load_hf(model_dir: &Path) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let text = std::fs::read_to_string(&config_path).map_err(|error| Error::Model {
            message: format!("Qwen3-MoE config '{}': {error}", config_path.display()),
        })?;
        let config_json = serde_json::from_str::<serde_json::Value>(&text).map_err(|source| {
            Error::ModelSource {
                source: Box::new(source),
            }
        })?;
        let config = Qwen3MoeConfig::from_value(&config_json)?;
        let recipe = Qwen3MoeRecipe::new();
        let checkpoint = DecoderLoadOptions::new(&recipe, &config_json)
            .open_hf_checkpoint(model_dir, ModelFamily::QwenMoe)?;
        let tokenizer = TokenizerHandle::load(model_dir)?;
        Ok(Self {
            config,
            checkpoint,
            tokenizer,
        })
    }

    pub const fn resources(&self) -> &BoundDecoderResources {
        self.checkpoint.resources()
    }

    pub(super) fn into_runtime_parts(self) -> (BoundDecoderResources, TokenizerHandle) {
        (self.checkpoint.into_resources(), self.tokenizer)
    }
}
