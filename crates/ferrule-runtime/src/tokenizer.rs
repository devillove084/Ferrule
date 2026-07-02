//! Lightweight tokenizer handle — decoupled from model weights.
//!
//! Uses `OlmoeModel::load_lightweight` to load only config + tokenizer
//! (skipping all large weight tensors), then extracts the tokenizer.

use ferrule_core::{Error, Result};
use ferrule_model::OlmoeModel;
use std::path::Path;

/// Lightweight tokenizer handle — decoupled from model weights.
pub struct TokenizerHandle {
    inner: tokenizers::Tokenizer,
    eos_token_id: Option<u32>,
}

impl TokenizerHandle {
    #[cfg(test)]
    pub(crate) fn from_parts(inner: tokenizers::Tokenizer, eos_token_id: Option<u32>) -> Self {
        Self {
            inner,
            eos_token_id,
        }
    }

    /// Load only tokenizer/config metadata from `model_dir`.
    ///
    /// Prefer the generic `tokenizer.json` path so non-OLMoE families do not have
    /// to pass through an OLMoE lightweight model loader. The OLMoE fallback is
    /// retained for legacy fixtures that rely on existing model loading behavior.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        if tokenizer_path.exists() {
            let inner = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
                Error::Tokenization(format!("tokenizer '{}': {e}", tokenizer_path.display()))
            })?;
            return Ok(Self {
                inner,
                eos_token_id: read_eos_token_id(model_dir)?,
            });
        }

        let model = OlmoeModel::load_lightweight(model_dir)?;
        let eos_token_id = model.config.eos_token_id;
        Ok(Self {
            inner: model.into_tokenizer(),
            eos_token_id,
        })
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.inner
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| ferrule_core::Error::Tokenization(format!("encode: {e}")))
    }

    /// Decode token IDs into text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| ferrule_core::Error::Tokenization(format!("decode: {e}")))
    }

    /// Return the EOS token ID from config, if set.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }
}

fn read_eos_token_id(model_dir: &Path) -> Result<Option<u32>> {
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&config_path)
        .map_err(|e| Error::Model(format!("config '{}': {e}", config_path.display())))?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| Error::Model(format!("config json '{}': {e}", config_path.display())))?;
    Ok(json
        .get("eos_token_id")
        .and_then(|value| value.as_u64())
        .map(|value| value as u32))
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_handle_eos() {
        // Create a minimal BPE tokenizer.
        let bpe = tokenizers::models::bpe::BPE::default();
        let mut tok = tokenizers::Tokenizer::new(bpe);
        tok.add_special_tokens(&[tokenizers::AddedToken::from("<unk>", true)]);
        let handle = TokenizerHandle {
            inner: tok,
            eos_token_id: Some(2),
        };
        assert_eq!(handle.eos_token_id(), Some(2));
    }

    #[test]
    fn tokenizer_handle_encode_decode_roundtrip() {
        // Build a minimal tokenizer that can encode/decode.
        let bpe = tokenizers::models::bpe::BPE::default();
        let mut tok = tokenizers::Tokenizer::new(bpe);
        tok.add_special_tokens(&[tokenizers::AddedToken::from("<unk>", true)]);
        // Add some basic tokens
        tok.add_tokens(&[tokenizers::AddedToken::from("hello", true)]);

        let handle = TokenizerHandle {
            inner: tok,
            eos_token_id: Some(2),
        };

        let encoded = handle.encode("hello").unwrap();
        // With our minimal BPE, encoding should produce tokens
        assert!(!encoded.is_empty());
        // Decoding should not panic
        let _decoded = handle.decode(&encoded).unwrap();
    }
}
