//! Generic speculative decoding boundary.
//!
//! DSpark, MTP, Eagle-style, or any future draft model should plug in here as a
//! draft/target policy. The base Transformer forward path remains unaware of the
//! specific speculation method.

use ferrule_core::Result;
use ferrule_model::{SpeculationMode, SpeculationPolicy as ModelSpeculationPolicy};

pub trait DraftModel {
    fn propose(&mut self, context: &[u32], max_tokens: usize) -> Result<Vec<u32>>;
}

pub trait TargetModel {
    fn accept(&mut self, context: &[u32], candidate: u32) -> Result<bool>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeDecodingPolicy {
    pub max_draft_tokens: usize,
    pub rollback_on_reject: bool,
}

impl SpeculativeDecodingPolicy {
    pub fn from_model_policy(policy: &ModelSpeculationPolicy, default_block_size: usize) -> Self {
        match policy.mode {
            SpeculationMode::None => Self::disabled(),
            SpeculationMode::DraftModel | SpeculationMode::MultiTokenPrediction => {
                Self::draft(default_block_size)
            }
        }
    }

    pub fn disabled() -> Self {
        Self {
            max_draft_tokens: 0,
            rollback_on_reject: true,
        }
    }

    pub fn draft(max_draft_tokens: usize) -> Self {
        Self {
            max_draft_tokens,
            rollback_on_reject: true,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.max_draft_tokens > 0
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpeculationMetrics {
    pub proposed_tokens: usize,
    pub accepted_tokens: usize,
    pub rejected_tokens: usize,
    pub rollbacks: usize,
    pub draft_invocations: usize,
    pub target_verifications: usize,
}

impl SpeculationMetrics {
    pub fn acceptance_rate(&self) -> f32 {
        if self.proposed_tokens == 0 {
            0.0
        } else {
            self.accepted_tokens as f32 / self.proposed_tokens as f32
        }
    }

    pub fn verification_speedup_ratio(&self) -> Option<f32> {
        (self.target_verifications > 0)
            .then(|| self.proposed_tokens as f32 / self.target_verifications as f32)
    }

    pub fn slowed_down(&self) -> bool {
        self.verification_speedup_ratio()
            .map(|ratio| ratio < 1.0)
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeStepOutput {
    pub accepted: Vec<u32>,
    pub rejected: Option<u32>,
    pub metrics: SpeculationMetrics,
}

pub fn run_speculative_step(
    context: &[u32],
    policy: SpeculativeDecodingPolicy,
    draft: &mut impl DraftModel,
    target: &mut impl TargetModel,
) -> Result<SpeculativeStepOutput> {
    if !policy.is_enabled() {
        return Ok(SpeculativeStepOutput {
            accepted: Vec::new(),
            rejected: None,
            metrics: SpeculationMetrics::default(),
        });
    }

    let proposals = draft.propose(context, policy.max_draft_tokens)?;
    let mut metrics = SpeculationMetrics {
        proposed_tokens: proposals.len(),
        draft_invocations: 1,
        ..SpeculationMetrics::default()
    };
    let mut accepted = Vec::new();
    let mut verify_context = context.to_vec();

    for candidate in proposals {
        metrics.target_verifications += 1;
        if target.accept(&verify_context, candidate)? {
            accepted.push(candidate);
            verify_context.push(candidate);
            metrics.accepted_tokens += 1;
        } else {
            metrics.rejected_tokens += 1;
            if policy.rollback_on_reject {
                metrics.rollbacks += 1;
            }
            return Ok(SpeculativeStepOutput {
                accepted,
                rejected: Some(candidate),
                metrics,
            });
        }
    }

    Ok(SpeculativeStepOutput {
        accepted,
        rejected: None,
        metrics,
    })
}

#[cfg(test)]
mod tests {
    use ferrule_core::Result;

    use super::*;

    #[test]
    fn speculation_policy_accepts_rejects_and_tracks_rollback() {
        let mut draft = FakeDraft {
            proposals: vec![10, 11, 12],
        };
        let mut target = FakeTarget {
            accepted_prefix: vec![10, 11],
        };
        let output = run_speculative_step(
            &[1, 2, 3],
            SpeculativeDecodingPolicy::draft(4),
            &mut draft,
            &mut target,
        )
        .unwrap();
        assert_eq!(output.accepted, vec![10, 11]);
        assert_eq!(output.rejected, Some(12));
        assert_eq!(output.metrics.proposed_tokens, 3);
        assert_eq!(output.metrics.accepted_tokens, 2);
        assert_eq!(output.metrics.rejected_tokens, 1);
        assert_eq!(output.metrics.rollbacks, 1);
        assert_eq!(output.metrics.draft_invocations, 1);
        assert_eq!(output.metrics.target_verifications, 3);
        assert_eq!(output.metrics.verification_speedup_ratio(), Some(1.0));
        assert!((output.metrics.acceptance_rate() - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn disabled_speculation_does_not_call_draft() {
        let mut draft = FakeDraft {
            proposals: vec![10, 11],
        };
        let mut target = FakeTarget {
            accepted_prefix: vec![10, 11],
        };
        let output = run_speculative_step(
            &[1],
            SpeculativeDecodingPolicy::disabled(),
            &mut draft,
            &mut target,
        )
        .unwrap();
        assert!(output.accepted.is_empty());
        assert_eq!(output.metrics.proposed_tokens, 0);
        assert_eq!(output.metrics.draft_invocations, 0);
    }

    #[test]
    fn speculation_policy_bridges_from_model_policy_without_dspark_forward_branch() {
        let policy = ModelSpeculationPolicy {
            mode: SpeculationMode::MultiTokenPrediction,
        };
        let runtime = SpeculativeDecodingPolicy::from_model_policy(&policy, 5);
        assert_eq!(runtime.max_draft_tokens, 5);
        assert!(runtime.rollback_on_reject);
    }

    struct FakeDraft {
        proposals: Vec<u32>,
    }

    impl DraftModel for FakeDraft {
        fn propose(&mut self, _context: &[u32], max_tokens: usize) -> Result<Vec<u32>> {
            Ok(self.proposals.iter().take(max_tokens).copied().collect())
        }
    }

    struct FakeTarget {
        accepted_prefix: Vec<u32>,
    }

    impl TargetModel for FakeTarget {
        fn accept(&mut self, context: &[u32], candidate: u32) -> Result<bool> {
            let offset = context.len().saturating_sub(3);
            Ok(self.accepted_prefix.get(offset).copied() == Some(candidate))
        }
    }
}
