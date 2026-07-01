//! Reference-result comparison utilities.
//!
//! This module deliberately does not run a model. It compares Ferrule-observed
//! tokenization/generation facts against a checked-in `ReferenceCommandManifest`,
//! allowing DSV4 bring-up to validate tokenizer and first-token behavior even when
//! the full target model is not available in CI.

use std::collections::BTreeSet;

use ferrule_core::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::reference_manifest::{PromptId, ReferenceCommandManifest, ReferenceTopKLogit};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceObservation {
    pub prompt_id: PromptId,
    pub token_ids: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub topk: Vec<ReferenceTopKLogit>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceComparisonReport {
    pub prompt_id: PromptId,
    pub tokenization_matches: bool,
    pub first_token_matches: Option<bool>,
    pub prefix_tokens_matched: usize,
    pub topk_overlap: Option<usize>,
    pub mismatches: Vec<ReferenceMismatch>,
}

impl ReferenceComparisonReport {
    pub fn is_match(&self) -> bool {
        self.mismatches.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferenceMismatch {
    Tokenization {
        expected: Vec<u32>,
        observed: Vec<u32>,
    },
    FirstToken {
        expected: u32,
        observed: Option<u32>,
    },
    GeneratedPrefix {
        index: usize,
        expected: u32,
        observed: Option<u32>,
    },
    TopKMissingToken {
        token: u32,
    },
}

pub fn compare_reference_observation(
    manifest: &ReferenceCommandManifest,
    observation: &ReferenceObservation,
) -> Result<ReferenceComparisonReport> {
    let golden = manifest.prompt(&observation.prompt_id).ok_or_else(|| {
        Error::Model(format!(
            "reference manifest '{}' does not contain prompt '{}'",
            manifest.id.as_str(),
            observation.prompt_id.as_str()
        ))
    })?;

    let mut mismatches = Vec::new();
    let tokenization_matches =
        golden.token_ids.is_empty() || golden.token_ids == observation.token_ids;
    if !tokenization_matches {
        mismatches.push(ReferenceMismatch::Tokenization {
            expected: golden.token_ids.clone(),
            observed: observation.token_ids.clone(),
        });
    }

    let first_token_matches = golden.expected_first_token.map(|expected| {
        let observed = observation.generated_tokens.first().copied();
        let matches = observed == Some(expected);
        if !matches {
            mismatches.push(ReferenceMismatch::FirstToken { expected, observed });
        }
        matches
    });

    let mut prefix_tokens_matched = 0usize;
    for (index, expected) in golden.expected_tokens.iter().copied().enumerate() {
        let observed = observation.generated_tokens.get(index).copied();
        if observed == Some(expected) {
            prefix_tokens_matched += 1;
        } else {
            mismatches.push(ReferenceMismatch::GeneratedPrefix {
                index,
                expected,
                observed,
            });
            break;
        }
    }

    let topk_overlap = if golden.reference_topk.is_empty() {
        None
    } else {
        let observed = observation
            .topk
            .iter()
            .map(|entry| entry.token)
            .collect::<BTreeSet<_>>();
        let mut overlap = 0usize;
        for entry in &golden.reference_topk {
            if observed.contains(&entry.token) {
                overlap += 1;
            } else {
                mismatches.push(ReferenceMismatch::TopKMissingToken { token: entry.token });
            }
        }
        Some(overlap)
    };

    Ok(ReferenceComparisonReport {
        prompt_id: observation.prompt_id.clone(),
        tokenization_matches,
        first_token_matches,
        prefix_tokens_matched,
        topk_overlap,
        mismatches,
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
    fn compare_reference_accepts_matching_first_token_and_topk() {
        let manifest = fixture_manifest();
        let report = compare_reference_observation(
            &manifest,
            &ReferenceObservation {
                prompt_id: PromptId::new("hello").unwrap(),
                token_ids: vec![10, 20],
                generated_tokens: vec![42, 43],
                topk: vec![
                    ReferenceTopKLogit {
                        token: 42,
                        logit: 3.0,
                    },
                    ReferenceTopKLogit {
                        token: 7,
                        logit: 1.0,
                    },
                ],
            },
        )
        .unwrap();
        assert!(report.is_match(), "{:?}", report.mismatches);
        assert_eq!(report.first_token_matches, Some(true));
        assert_eq!(report.prefix_tokens_matched, 2);
        assert_eq!(report.topk_overlap, Some(2));
    }

    #[test]
    fn compare_reference_reports_precise_mismatch() {
        let manifest = fixture_manifest();
        let report = compare_reference_observation(
            &manifest,
            &ReferenceObservation {
                prompt_id: PromptId::new("hello").unwrap(),
                token_ids: vec![10, 21],
                generated_tokens: vec![41],
                topk: vec![ReferenceTopKLogit {
                    token: 42,
                    logit: 3.0,
                }],
            },
        )
        .unwrap();
        assert!(!report.is_match());
        assert!(matches!(
            report.mismatches[0],
            ReferenceMismatch::Tokenization { .. }
        ));
        assert!(report.mismatches.iter().any(|mismatch| matches!(
            mismatch,
            ReferenceMismatch::FirstToken {
                expected: 42,
                observed: Some(41)
            }
        )));
    }

    fn fixture_manifest() -> ReferenceCommandManifest {
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
                expected_first_token: Some(42),
                expected_tokens: vec![42, 43],
                reference_topk: vec![
                    ReferenceTopKLogit {
                        token: 42,
                        logit: 3.0,
                    },
                    ReferenceTopKLogit {
                        token: 7,
                        logit: 1.0,
                    },
                ],
            }],
            notes: Vec::new(),
        }
    }
}
