//! Reference-engine manifests for correctness checks.
//!
//! Ferrule may not be able to execute a full target model while a new family is
//! being brought up. This manifest records how an external/reference engine should
//! be invoked and which token/logit facts Ferrule can compare against once the
//! tokenizer and first-token path are wired.

use ferrule_core::{Error, Result};
use ferrule_model::ModelFamily;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ReferenceManifestId(String);

impl ReferenceManifestId {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(Error::Model("reference manifest id is empty".into()));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PromptId(String);

impl PromptId {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(Error::Model("golden prompt id is empty".into()));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferenceArtifact {
    HuggingFaceRepo {
        repo_id: String,
        revision: Option<String>,
    },
    LocalPath {
        path: String,
    },
    GgufPath {
        path: String,
    },
    SyntheticFixture {
        fixture: String,
    },
}

impl ReferenceArtifact {
    pub fn deepseek_v4_flash_dspark_official() -> Self {
        Self::HuggingFaceRepo {
            repo_id: "deepseek-ai/DeepSeek-V4-Flash-DSpark".into(),
            revision: Some("913f0657a874f76844e2e91cbe706dbcaceeb6d7".into()),
        }
    }

    fn validate(&self, manifest_id: &ReferenceManifestId) -> Result<()> {
        let empty = match self {
            Self::HuggingFaceRepo { repo_id, .. } => repo_id.trim().is_empty(),
            Self::LocalPath { path } | Self::GgufPath { path } => path.trim().is_empty(),
            Self::SyntheticFixture { fixture } => fixture.trim().is_empty(),
        };
        if empty {
            return Err(Error::Model(format!(
                "reference manifest '{}' has an empty artifact locator",
                manifest_id.as_str()
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferenceEngineKind {
    LlamaCpp,
    OfficialRuntime,
    PythonFixture,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceCommand {
    pub program: String,
    pub args: Vec<String>,
}

impl ReferenceCommand {
    pub fn new(
        program: impl Into<String>,
        args: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            program: program.into(),
            args: args.into_iter().map(Into::into).collect(),
        }
    }

    pub fn argv(&self) -> Vec<&str> {
        std::iter::once(self.program.as_str())
            .chain(self.args.iter().map(String::as_str))
            .collect()
    }

    fn validate(&self, manifest_id: &ReferenceManifestId) -> Result<()> {
        if self.program.trim().is_empty() || self.args.iter().any(|part| part.trim().is_empty()) {
            return Err(Error::Model(format!(
                "reference manifest '{}' has an empty command part",
                manifest_id.as_str()
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceCommandManifest {
    pub id: ReferenceManifestId,
    pub family: ModelFamily,
    pub artifact: ReferenceArtifact,
    pub engine: ReferenceEngineKind,
    pub command: ReferenceCommand,
    pub prompts: Vec<GoldenPrompt>,
    pub notes: Vec<String>,
}

impl ReferenceCommandManifest {
    pub fn deepseek_v4_first_token_smoke() -> Self {
        Self {
            id: ReferenceManifestId::new("deepseek-v4-first-token-smoke").expect("static id"),
            family: ModelFamily::DeepSeekV4,
            artifact: ReferenceArtifact::deepseek_v4_flash_dspark_official(),
            engine: ReferenceEngineKind::OfficialRuntime,
            command: ReferenceCommand::new(
                "python",
                ["inference/generate.py", "--max-new-tokens", "1"],
            ),
            prompts: vec![GoldenPrompt {
                id: PromptId::new("hello").expect("static prompt id"),
                prompt: "hello".into(),
                token_ids: Vec::new(),
                expected_first_token: None,
                expected_tokens: Vec::new(),
                reference_topk: Vec::new(),
            }],
            notes: vec![
                "Fill token_ids/expected_first_token from the selected official runtime before claiming first-token support".into(),
            ],
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.artifact.validate(&self.id)?;
        self.command.validate(&self.id)?;
        if self.prompts.is_empty() {
            return Err(Error::Model(format!(
                "reference manifest '{}' must contain at least one prompt",
                self.id.as_str()
            )));
        }
        for prompt in &self.prompts {
            prompt.validate()?;
        }
        Ok(())
    }

    pub fn prompt(&self, id: &PromptId) -> Option<&GoldenPrompt> {
        self.prompts.iter().find(|prompt| &prompt.id == id)
    }

    pub fn first_token_ready(&self) -> bool {
        self.prompts
            .iter()
            .all(|prompt| prompt.expected_first_token.is_some())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenPrompt {
    pub id: PromptId,
    pub prompt: String,
    pub token_ids: Vec<u32>,
    pub expected_first_token: Option<u32>,
    pub expected_tokens: Vec<u32>,
    pub reference_topk: Vec<ReferenceTopKLogit>,
}

impl GoldenPrompt {
    fn validate(&self) -> Result<()> {
        if self.prompt.is_empty() && self.token_ids.is_empty() {
            return Err(Error::Model(format!(
                "golden prompt '{}' must include prompt text or token ids",
                self.id.as_str()
            )));
        }
        if let Some(first) = self.expected_first_token {
            if !self.expected_tokens.is_empty() && self.expected_tokens[0] != first {
                return Err(Error::Model(format!(
                    "golden prompt '{}' first token mismatch: expected_first_token={} but expected_tokens starts with {}",
                    self.id.as_str(), first, self.expected_tokens[0]
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceTopKLogit {
    pub token: u32,
    pub logit: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_manifest_stores_deepseek_first_token_plan() {
        let manifest = ReferenceCommandManifest::deepseek_v4_first_token_smoke();
        manifest.validate().unwrap();
        assert_eq!(manifest.family, ModelFamily::DeepSeekV4);
        assert_eq!(manifest.prompts.len(), 1);
        assert!(!manifest.first_token_ready());
        assert!(manifest
            .command
            .argv()
            .iter()
            .any(|part| *part == "inference/generate.py"));
    }

    #[test]
    fn reference_manifest_rejects_inconsistent_first_token() {
        let manifest = ReferenceCommandManifest {
            id: ReferenceManifestId::new("bad").unwrap(),
            family: ModelFamily::DeepSeekV4,
            artifact: ReferenceArtifact::SyntheticFixture {
                fixture: "fixture".into(),
            },
            engine: ReferenceEngineKind::PythonFixture,
            command: ReferenceCommand::new("python", ["fixture.py"]),
            prompts: vec![GoldenPrompt {
                id: PromptId::new("p").unwrap(),
                prompt: "x".into(),
                token_ids: vec![1],
                expected_first_token: Some(2),
                expected_tokens: vec![3],
                reference_topk: Vec::new(),
            }],
            notes: Vec::new(),
        };
        let err = manifest.validate().unwrap_err();
        assert!(err.to_string().contains("first token mismatch"));
    }
}
