//! Expert routing policies and CPU reference semantics.
//!
//! This module captures router math independently from any one model family or
//! backend kernel. Model-family adapters choose score functions, selection modes,
//! normalization, bias handling, and route scaling through this typed policy.

use std::cmp::Ordering;

use ferrule_common::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterScoreFunction {
    Softmax,
    Sigmoid,
    SqrtSoftplus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterSelectionPolicy {
    /// Select top-k experts by score, optionally after adding router bias.
    ScoreTopK,
    /// Select experts from an external token-id -> expert-id table. Scores are
    /// still computed and gathered to produce routing weights.
    Hash,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpertRouterPolicy {
    pub top_k: usize,
    pub score_function: RouterScoreFunction,
    pub selection: RouterSelectionPolicy,
    /// Normalize selected non-softmax scores before applying `route_scale`.
    /// Softmax scores are already normalized over all experts and are not
    /// renormalized by the reference gate.
    pub normalize_non_softmax_weights: bool,
    pub route_scale: f32,
}

impl ExpertRouterPolicy {
    pub fn sqrt_softplus_score_topk(top_k: usize, route_scale: f32) -> Self {
        Self {
            top_k,
            score_function: RouterScoreFunction::SqrtSoftplus,
            selection: RouterSelectionPolicy::ScoreTopK,
            normalize_non_softmax_weights: true,
            route_scale,
        }
    }

    pub fn sqrt_softplus_hash(top_k: usize, route_scale: f32) -> Self {
        Self {
            selection: RouterSelectionPolicy::Hash,
            ..Self::sqrt_softplus_score_topk(top_k, route_scale)
        }
    }

    pub fn route(
        &self,
        logits: &[f32],
        bias: Option<&[f32]>,
        hash_experts: Option<&[usize]>,
    ) -> Result<Vec<ExpertRoute>> {
        validate_policy(self, logits, bias, hash_experts)?;
        let original_scores = score_logits(logits, self.score_function)?;
        let indices = match self.selection {
            RouterSelectionPolicy::ScoreTopK => select_score_topk(
                &original_scores,
                bias,
                self.top_k,
                self.score_function == RouterScoreFunction::Softmax,
            ),
            RouterSelectionPolicy::Hash => hash_experts
                .expect("validated")
                .iter()
                .take(self.top_k)
                .copied()
                .collect::<Vec<_>>(),
        };

        let mut weights = indices
            .iter()
            .map(|&expert| original_scores[expert])
            .collect::<Vec<_>>();
        if self.score_function != RouterScoreFunction::Softmax && self.normalize_non_softmax_weights
        {
            let sum = weights.iter().sum::<f32>();
            if sum <= 0.0 || !sum.is_finite() {
                return Err(Error::Model(format!(
                    "router selected non-positive or non-finite weight sum: {sum}"
                )));
            }
            for weight in &mut weights {
                *weight /= sum;
            }
        }
        for weight in &mut weights {
            *weight *= self.route_scale;
        }

        Ok(indices
            .into_iter()
            .zip(weights)
            .map(|(expert, weight)| ExpertRoute {
                expert,
                weight,
                score: original_scores[expert],
                selection_score: original_scores[expert] + bias.map(|b| b[expert]).unwrap_or(0.0),
            })
            .collect())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpertRoute {
    pub expert: usize,
    pub weight: f32,
    /// Score before router bias. Routing weights are gathered from this value.
    pub score: f32,
    /// Score used for score-top-k selection. For hash routing this is diagnostic
    /// only; the hash table chooses the expert ids.
    pub selection_score: f32,
}

fn validate_policy(
    policy: &ExpertRouterPolicy,
    logits: &[f32],
    bias: Option<&[f32]>,
    hash_experts: Option<&[usize]>,
) -> Result<()> {
    if logits.is_empty() {
        return Err(Error::Model("router logits are empty".into()));
    }
    if policy.top_k == 0 || policy.top_k > logits.len() {
        return Err(Error::Model(format!(
            "router top_k must be in 1..={}, got {}",
            logits.len(),
            policy.top_k
        )));
    }
    if !policy.route_scale.is_finite() {
        return Err(Error::Model(format!(
            "router route_scale must be finite, got {}",
            policy.route_scale
        )));
    }
    if let Some(bias) = bias {
        if bias.len() != logits.len() {
            return Err(Error::Model(format!(
                "router bias length mismatch: expected {}, got {}",
                logits.len(),
                bias.len()
            )));
        }
    }
    for (idx, value) in logits.iter().enumerate() {
        if !value.is_finite() {
            return Err(Error::Model(format!(
                "router logit at expert {idx} is not finite: {value}"
            )));
        }
    }
    if let Some(bias) = bias {
        for (idx, value) in bias.iter().enumerate() {
            if !value.is_finite() {
                return Err(Error::Model(format!(
                    "router bias at expert {idx} is not finite: {value}"
                )));
            }
        }
    }
    if policy.selection == RouterSelectionPolicy::Hash {
        let hash_experts = hash_experts
            .ok_or_else(|| Error::Model("hash router requires token-id expert indices".into()))?;
        if hash_experts.len() < policy.top_k {
            return Err(Error::Model(format!(
                "hash router requires at least {} expert ids, got {}",
                policy.top_k,
                hash_experts.len()
            )));
        }
        for (rank, &expert) in hash_experts.iter().take(policy.top_k).enumerate() {
            if expert >= logits.len() {
                return Err(Error::Model(format!(
                    "hash router expert id {expert} exceeds expert count {}",
                    logits.len()
                )));
            }
            if hash_experts[..rank].contains(&expert) {
                return Err(Error::Model(format!(
                    "hash router selected duplicate expert id {expert} within the first {} routes",
                    policy.top_k
                )));
            }
        }
    }
    Ok(())
}

fn score_logits(logits: &[f32], function: RouterScoreFunction) -> Result<Vec<f32>> {
    match function {
        RouterScoreFunction::Softmax => Ok(softmax(logits)),
        RouterScoreFunction::Sigmoid => Ok(logits.iter().map(|&value| sigmoid(value)).collect()),
        RouterScoreFunction::SqrtSoftplus => Ok(logits
            .iter()
            .map(|&value| stable_softplus(value).sqrt())
            .collect()),
    }
}

fn select_score_topk(
    scores: &[f32],
    bias: Option<&[f32]>,
    top_k: usize,
    scores_are_softmax: bool,
) -> Vec<usize> {
    let mut ranked = scores
        .iter()
        .enumerate()
        .map(|(expert, &score)| {
            let selection_score = if scores_are_softmax {
                score
            } else {
                score + bias.map(|b| b[expert]).unwrap_or(0.0)
            };
            (expert, selection_score)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|(left_idx, left_score), (right_idx, right_score)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| left_idx.cmp(right_idx))
    });
    ranked.truncate(top_k);
    ranked.into_iter().map(|(expert, _)| expert).collect()
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, value| acc.max(value));
    let mut values = logits
        .iter()
        .map(|&value| (value - max).exp())
        .collect::<Vec<_>>();
    let sum = values.iter().sum::<f32>();
    if sum > 0.0 {
        for value in &mut values {
            *value /= sum;
        }
    }
    values
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn stable_softplus(value: f32) -> f32 {
    match value.total_cmp(&20.0) {
        Ordering::Greater => value,
        _ => value.exp().ln_1p(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqrt_softplus_router_normalizes_and_scales_weights() {
        let policy = ExpertRouterPolicy::sqrt_softplus_score_topk(2, 1.5);
        let routes = policy.route(&[0.0, 1.0, -1.0], None, None).unwrap();
        assert_eq!(routes[0].expert, 1);
        assert_eq!(routes[1].expert, 0);

        let score_1 = stable_softplus(1.0).sqrt();
        let score_0 = stable_softplus(0.0).sqrt();
        let denom = score_1 + score_0;
        assert_close(routes[0].weight, score_1 / denom * 1.5);
        assert_close(routes[1].weight, score_0 / denom * 1.5);
    }

    #[test]
    fn bias_changes_score_topk_selection_but_not_gathered_scores() {
        let policy = ExpertRouterPolicy::sqrt_softplus_score_topk(2, 1.0);
        let routes = policy
            .route(&[5.0, 4.0, -4.0], Some(&[0.0, 0.0, 10.0]), None)
            .unwrap();
        assert_eq!(routes[0].expert, 2);
        assert_eq!(routes[1].expert, 0);
        assert_close(routes[0].score, stable_softplus(-4.0).sqrt());
        assert_close(
            routes[0].selection_score,
            stable_softplus(-4.0).sqrt() + 10.0,
        );
    }

    #[test]
    fn hash_router_preserves_hash_indices_and_gathers_original_scores() {
        let policy = ExpertRouterPolicy::sqrt_softplus_hash(2, 1.5);
        let routes = policy.route(&[0.0, 1.0, 2.0], None, Some(&[2, 0])).unwrap();
        assert_eq!(
            routes.iter().map(|route| route.expert).collect::<Vec<_>>(),
            vec![2, 0]
        );

        let score_2 = stable_softplus(2.0).sqrt();
        let score_0 = stable_softplus(0.0).sqrt();
        let denom = score_2 + score_0;
        assert_close(routes[0].weight, score_2 / denom * 1.5);
        assert_close(routes[1].weight, score_0 / denom * 1.5);
    }

    #[test]
    fn softmax_router_does_not_renormalize_selected_weights() {
        let policy = ExpertRouterPolicy {
            top_k: 1,
            score_function: RouterScoreFunction::Softmax,
            selection: RouterSelectionPolicy::ScoreTopK,
            normalize_non_softmax_weights: true,
            route_scale: 1.0,
        };
        let routes = policy
            .route(&[0.0, 1.0], Some(&[100.0, 0.0]), None)
            .unwrap();
        assert_eq!(routes[0].expert, 1);
        assert!(routes[0].weight < 1.0);
        assert_close(routes[0].weight, 1.0f32.exp() / (1.0 + 1.0f32.exp()));
    }

    #[test]
    fn hash_router_rejects_out_of_range_expert_ids() {
        let policy = ExpertRouterPolicy::sqrt_softplus_hash(1, 1.0);
        let err = policy.route(&[0.0, 1.0], None, Some(&[2])).unwrap_err();
        assert!(err.to_string().contains("exceeds expert count"));
    }

    #[test]
    fn hash_router_rejects_duplicate_selected_experts() {
        let policy = ExpertRouterPolicy::sqrt_softplus_hash(2, 1.0);
        let err = policy
            .route(&[0.0, 1.0, 2.0], None, Some(&[1, 1, 2]))
            .unwrap_err();
        assert!(err.to_string().contains("duplicate expert id 1"));
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1e-6, "{actual} vs {expected}");
    }
}
