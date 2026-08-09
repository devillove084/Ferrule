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
            return Err(Error::Tokenization {
                message: "incremental decode produced a non-prefix continuation".into(),
            });
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
    eos_token_ids: Vec<u32>,
}

impl TokenizerHandle {
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn from_parts(inner: tokenizers::Tokenizer, eos_token_id: Option<u32>) -> Self {
        Self {
            inner,
            eos_token_ids: eos_token_id.into_iter().collect(),
        }
    }

    /// Load tokenizer and EOS config from a HuggingFace model directory.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        let inner =
            tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| Error::Tokenization {
                message: format!("tokenizer '{}': {e}", tokenizer_path.display()),
            })?;
        Ok(Self {
            inner,
            eos_token_ids: read_eos_token_ids(model_dir)?,
        })
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.inner
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| ferrule_common::Error::Tokenization {
                message: format!("encode: {e}"),
            })
    }

    /// Decode token IDs into text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| ferrule_common::Error::Tokenization {
                message: format!("decode: {e}"),
            })
    }

    /// Return the first configured EOS token ID, preserving the legacy API.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_ids.first().copied()
    }

    /// Return every configured EOS token ID in configuration order.
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    /// Return whether `token_id` is any configured EOS token.
    pub fn is_eos_token(&self, token_id: u32) -> bool {
        self.eos_token_ids.contains(&token_id)
    }
}

fn read_eos_token_ids(model_dir: &Path) -> Result<Vec<u32>> {
    for filename in ["generation_config.json", "config.json"] {
        let config_path = model_dir.join(filename);
        if !config_path.exists() {
            continue;
        }
        let text = std::fs::read_to_string(&config_path).map_err(|e| Error::Model {
            message: format!("config '{}': {e}", config_path.display()),
        })?;
        let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| Error::Model {
            message: format!("config json '{}': {e}", config_path.display()),
        })?;
        let Some(value) = json.get("eos_token_id") else {
            continue;
        };
        let ids = parse_eos_token_ids(value, &config_path)?;
        if !ids.is_empty() {
            return Ok(ids);
        }
    }
    Ok(Vec::new())
}

fn parse_eos_token_ids(value: &serde_json::Value, path: &Path) -> Result<Vec<u32>> {
    let values = match value {
        serde_json::Value::Null => return Ok(Vec::new()),
        serde_json::Value::Number(_) => std::slice::from_ref(value),
        serde_json::Value::Array(values) => values,
        _ => {
            return Err(Error::Model {
                message: format!("config '{}' has non-integer eos_token_id", path.display()),
            });
        }
    };

    let mut ids = Vec::with_capacity(values.len());
    for value in values {
        let id = value
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| Error::Model {
                message: format!(
                    "config '{}' has eos_token_id outside the u32 range",
                    path.display()
                ),
            })?;
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    Ok(ids)
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TempModelDir(PathBuf);

    impl TempModelDir {
        fn new(name: &str) -> Self {
            static NEXT_ID: AtomicU64 = AtomicU64::new(0);
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "ferrule-tokenizer-{name}-{}-{id}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn write(&self, filename: &str, contents: &str) {
            std::fs::write(self.0.join(filename), contents).unwrap();
        }
    }

    impl Drop for TempModelDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    impl AsRef<Path> for TempModelDir {
        fn as_ref(&self) -> &Path {
            &self.0
        }
    }

    #[test]
    fn tokenizer_handle_eos() {
        let bpe = tokenizers::models::bpe::BPE::default();
        let mut tok = tokenizers::Tokenizer::new(bpe);
        tok.add_special_tokens([tokenizers::AddedToken::from("<unk>", true)])
            .expect("add test special token");
        let handle = TokenizerHandle {
            inner: tok,
            eos_token_ids: vec![2, 3],
        };
        assert_eq!(handle.eos_token_id(), Some(2));
        assert_eq!(handle.eos_token_ids(), [2, 3]);
        assert!(handle.is_eos_token(2));
        assert!(handle.is_eos_token(3));
        assert!(!handle.is_eos_token(4));
    }

    #[test]
    fn generation_config_array_takes_priority_and_deduplicates_eos_tokens() {
        let model_dir = TempModelDir::new("generation-array");
        model_dir.write(
            "generation_config.json",
            r#"{"eos_token_id":[151645,151643,151645]}"#,
        );
        model_dir.write("config.json", r#"{"eos_token_id":2}"#);

        assert_eq!(
            read_eos_token_ids(model_dir.as_ref()).unwrap(),
            [151645, 151643]
        );
    }

    #[test]
    fn generation_config_accepts_integer_eos_token() {
        let model_dir = TempModelDir::new("generation-integer");
        model_dir.write("generation_config.json", r#"{"eos_token_id":151645}"#);

        assert_eq!(read_eos_token_ids(model_dir.as_ref()).unwrap(), [151645]);
    }

    #[test]
    fn config_json_is_the_fallback_for_missing_or_empty_generation_eos() {
        for generation_config in [
            r#"{"temperature":0.6}"#,
            r#"{"eos_token_id":[]}"#,
            r#"{"eos_token_id":null}"#,
        ] {
            let model_dir = TempModelDir::new("config-fallback");
            model_dir.write("generation_config.json", generation_config);
            model_dir.write("config.json", r#"{"eos_token_id":[7,8,7]}"#);

            assert_eq!(read_eos_token_ids(model_dir.as_ref()).unwrap(), [7, 8]);
        }
    }

    #[test]
    fn invalid_generation_eos_is_reported_instead_of_silently_falling_back() {
        let model_dir = TempModelDir::new("invalid-generation");
        model_dir.write("generation_config.json", r#"{"eos_token_id":[1,"two"]}"#);
        model_dir.write("config.json", r#"{"eos_token_id":2}"#);

        let error = read_eos_token_ids(model_dir.as_ref())
            .unwrap_err()
            .to_string();
        assert!(error.contains("generation_config.json"));
        assert!(error.contains("u32 range"));
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
        tok.add_special_tokens([tokenizers::AddedToken::from("hello", true)])
            .expect("add test special token");

        let handle = TokenizerHandle {
            inner: tok,
            eos_token_ids: vec![2],
        };

        let encoded = handle.encode("hello").unwrap();
        assert!(!encoded.is_empty());
        let _decoded = handle.decode(&encoded).unwrap();
    }
}
