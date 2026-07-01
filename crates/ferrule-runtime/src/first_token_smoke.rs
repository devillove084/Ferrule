//! First-token smoke harness.
//!
//! This is intentionally model-runner agnostic. A full Ferrule runner, a sliced
//! DSV4 fixture, or a fake test runner can implement `FirstTokenModel`; the harness
//! handles reference-manifest lookup and comparison without requiring a local full
//! DeepSeek V4 checkpoint.

use ferrule_core::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::reference_compare::{compare_reference_observation, ReferenceObservation};
use crate::reference_manifest::{PromptId, ReferenceCommandManifest};

pub trait FirstTokenModel {
    fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>>;
    fn generate_first_token(&mut self, prompt_tokens: &[u32]) -> Result<Option<u32>>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FirstTokenSmokeStatus {
    Matched,
    Mismatched,
    ReferenceIncomplete,
    Unsupported(FirstTokenUnsupportedReason),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FirstTokenUnsupportedReason {
    MissingPrompt,
    MissingExpectedFirstToken,
    ModelCannotExecute(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FirstTokenSmokeReport {
    pub prompt_id: PromptId,
    pub status: FirstTokenSmokeStatus,
    pub prompt_tokens: Vec<u32>,
    pub generated_first_token: Option<u32>,
}

impl FirstTokenSmokeReport {
    pub fn is_match(&self) -> bool {
        self.status == FirstTokenSmokeStatus::Matched
    }
}

pub fn run_first_token_smoke(
    manifest: &ReferenceCommandManifest,
    prompt_id: &PromptId,
    model: &mut impl FirstTokenModel,
) -> Result<FirstTokenSmokeReport> {
    let Some(prompt) = manifest.prompt(prompt_id) else {
        return Ok(FirstTokenSmokeReport {
            prompt_id: prompt_id.clone(),
            status: FirstTokenSmokeStatus::Unsupported(FirstTokenUnsupportedReason::MissingPrompt),
            prompt_tokens: Vec::new(),
            generated_first_token: None,
        });
    };
    if prompt.expected_first_token.is_none() {
        return Ok(FirstTokenSmokeReport {
            prompt_id: prompt_id.clone(),
            status: FirstTokenSmokeStatus::ReferenceIncomplete,
            prompt_tokens: prompt.token_ids.clone(),
            generated_first_token: None,
        });
    }

    let prompt_tokens = model.encode_prompt(&prompt.prompt).map_err(|err| {
        Error::Model(format!(
            "first-token smoke prompt '{}' encode failed: {err}",
            prompt_id.as_str()
        ))
    })?;
    let generated_first_token = model.generate_first_token(&prompt_tokens).map_err(|err| {
        Error::Model(format!(
            "first-token smoke prompt '{}' execution failed: {err}",
            prompt_id.as_str()
        ))
    })?;
    let observation = ReferenceObservation {
        prompt_id: prompt_id.clone(),
        token_ids: prompt_tokens.clone(),
        generated_tokens: generated_first_token.into_iter().collect(),
        topk: Vec::new(),
    };
    let comparison = compare_reference_observation(manifest, &observation)?;
    Ok(FirstTokenSmokeReport {
        prompt_id: prompt_id.clone(),
        status: if comparison.is_match() {
            FirstTokenSmokeStatus::Matched
        } else {
            FirstTokenSmokeStatus::Mismatched
        },
        prompt_tokens,
        generated_first_token,
    })
}

#[cfg(test)]
mod tests {
    use ferrule_model::ModelFamily;

    use super::*;
    use crate::reference_manifest::{
        GoldenPrompt, ReferenceArtifact, ReferenceCommand, ReferenceCommandManifest,
        ReferenceEngineKind, ReferenceManifestId,
    };

    #[test]
    fn first_token_smoke_matches_fake_sliced_fixture() {
        let manifest = fixture_manifest(Some(42));
        let mut model = FakeFirstTokenModel { first: Some(42) };
        let report =
            run_first_token_smoke(&manifest, &PromptId::new("hello").unwrap(), &mut model).unwrap();
        assert!(report.is_match(), "{report:?}");
        assert_eq!(report.prompt_tokens, vec![10, 20]);
        assert_eq!(report.generated_first_token, Some(42));
    }

    #[test]
    fn first_token_smoke_reports_reference_incomplete_without_model_execution() {
        let manifest = fixture_manifest(None);
        let mut model = FakeFirstTokenModel { first: Some(42) };
        let report =
            run_first_token_smoke(&manifest, &PromptId::new("hello").unwrap(), &mut model).unwrap();
        assert_eq!(report.status, FirstTokenSmokeStatus::ReferenceIncomplete);
        assert_eq!(report.generated_first_token, None);
    }

    struct FakeFirstTokenModel {
        first: Option<u32>,
    }

    impl FirstTokenModel for FakeFirstTokenModel {
        fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
            assert_eq!(prompt, "hello");
            Ok(vec![10, 20])
        }

        fn generate_first_token(&mut self, _prompt_tokens: &[u32]) -> Result<Option<u32>> {
            Ok(self.first)
        }
    }

    fn fixture_manifest(first: Option<u32>) -> ReferenceCommandManifest {
        ReferenceCommandManifest {
            id: ReferenceManifestId::new("fixture").unwrap(),
            family: ModelFamily::DeepSeekV4,
            artifact: ReferenceArtifact::SyntheticFixture {
                fixture: "tiny-dsv4".into(),
            },
            engine: ReferenceEngineKind::PythonFixture,
            command: ReferenceCommand::new("python", ["fixture.py"]),
            prompts: vec![GoldenPrompt {
                id: PromptId::new("hello").unwrap(),
                prompt: "hello".into(),
                token_ids: vec![10, 20],
                expected_first_token: first,
                expected_tokens: first.into_iter().collect(),
                reference_topk: Vec::new(),
            }],
            notes: Vec::new(),
        }
    }
}
