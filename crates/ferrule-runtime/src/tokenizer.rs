//! Lightweight tokenizer handle — decoupled from model weights.
//!
//! Uses `OlmoeModel::load_lightweight` to load only config + tokenizer
//! (skipping all large weight tensors), then extracts the tokenizer.

use ferrule_core::Result;
use ferrule_model::OlmoeModel;
use std::path::Path;

/// Lightweight tokenizer handle — decoupled from model weights.
pub struct TokenizerHandle {
    inner: tokenizers::Tokenizer,
    eos_token_id: Option<u32>,
}

impl TokenizerHandle {
    /// Load only the tokenizer and config from `model_dir`.
    /// Uses lightweight model loading (~1s instead of full weight load).
    pub fn load(model_dir: &Path) -> Result<Self> {
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
