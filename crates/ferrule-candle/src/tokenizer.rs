use ferrule_core::{FerruleError, FerruleResult};
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct FerruleTokenizer {
    inner: Tokenizer,
}

impl FerruleTokenizer {
    pub fn from_file(path: &Path) -> FerruleResult<Self> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| FerruleError::Model(format!("failed to load tokenizer: {e}")))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> FerruleResult<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| FerruleError::Model(format!("tokenizer encode failed: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> FerruleResult<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| FerruleError::Model(format!("tokenizer decode failed: {e}")))
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn vocab_size_hint(&self) -> usize {
        self.inner.get_vocab_size(false)
    }
}
