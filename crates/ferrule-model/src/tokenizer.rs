//! Lightweight tokenizer handle — decoupled from model weights.

use ferrule_common::{Error, Result};
use std::path::Path;

/// Per-sequence state for bounded incremental token decoding.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IncrementalDecodeState {
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl IncrementalDecodeState {
    pub fn reset(&mut self) {
        self.ids.clear();
        self.prefix.clear();
        self.prefix_index = 0;
    }

    /// Decode one token while retaining only the bounded context needed by the
    /// tokenizer decoder. `None` means more token bytes are required before a
    /// valid text delta can be emitted.
    pub fn step(
        &mut self,
        id: u32,
        decode: impl Fn(&[u32]) -> Result<String>,
    ) -> Result<Option<String>> {
        self.ids.push(id);
        let decoded = decode(&self.ids)?;
        if decoded.len() <= self.prefix.len() || decoded.ends_with('�') {
            return Ok(None);
        }
        if !decoded.starts_with(&self.prefix) {
            return Err(Error::Tokenization(
                "incremental decode produced a non-prefix continuation".into(),
            ));
        }

        let delta = decoded[self.prefix.len()..].to_owned();
        let new_prefix_index = self.ids.len().saturating_sub(self.prefix_index);
        self.ids = self.ids.drain(self.prefix_index..).collect();
        self.prefix = decode(&self.ids)?;
        self.prefix_index = new_prefix_index;
        Ok(Some(delta))
    }
}

/// Lightweight tokenizer handle — decoupled from model weights.
pub struct TokenizerHandle {
    inner: tokenizers::Tokenizer,
    eos_token_id: Option<u32>,
}

impl TokenizerHandle {
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn from_parts(inner: tokenizers::Tokenizer, eos_token_id: Option<u32>) -> Self {
        Self {
            inner,
            eos_token_id,
        }
    }

    /// Load tokenizer and EOS config from a HuggingFace model directory.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            Error::Tokenization(format!("tokenizer '{}': {e}", tokenizer_path.display()))
        })?;
        Ok(Self {
            inner,
            eos_token_id: read_eos_token_id(model_dir)?,
        })
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.inner
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| ferrule_common::Error::Tokenization(format!("encode: {e}")))
    }

    /// Decode token IDs into text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| ferrule_common::Error::Tokenization(format!("decode: {e}")))
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
    fn incremental_decode_waits_for_valid_text_and_emits_only_the_delta() {
        let mut state = IncrementalDecodeState::default();
        let decode = |ids: &[u32]| match ids {
            [1] => Ok("�".to_owned()),
            [1, 2] => Ok("é".to_owned()),
            [1, 2, 3] => Ok("é!".to_owned()),
            [2, 3] => Ok("é!".to_owned()),
            [3] => Ok("!".to_owned()),
            _ => Ok(String::new()),
        };

        assert_eq!(state.step(1, decode).unwrap(), None);
        assert_eq!(state.step(2, decode).unwrap(), Some("é".to_owned()));
        assert_eq!(state.step(3, decode).unwrap(), Some("!".to_owned()));
        state.reset();
        assert_eq!(state, IncrementalDecodeState::default());
    }

    #[test]
    fn tokenizer_handle_encode_decode_roundtrip() {
        let bpe = tokenizers::models::bpe::BPE::default();
        let mut tok = tokenizers::Tokenizer::new(bpe);
        tok.add_special_tokens(&[tokenizers::AddedToken::from("hello", true)]);

        let handle = TokenizerHandle {
            inner: tok,
            eos_token_id: Some(2),
        };

        let encoded = handle.encode("hello").unwrap();
        assert!(!encoded.is_empty());
        let _decoded = handle.decode(&encoded).unwrap();
    }
}
