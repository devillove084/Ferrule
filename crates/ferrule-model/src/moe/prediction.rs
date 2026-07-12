//! Expert hotset prediction and feedback-driven prefetch scoring.
//!
//! This module is deliberately model/backend agnostic. It does not know how an
//! expert is stored or materialized; it only consumes routing/residency feedback
//! and produces ranked cache actions. Backends remain free to interpret those
//! actions as GPU upload, host staging, or no-op keep-resident decisions.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::moe::routing::ExpertRoute;
use crate::moe::streaming::{ExpertId, ExpertLoadReason, ExpertStreamingStep};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertAccessPhase {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertResidency {
    Cold,
    HostStaged,
    Materializing,
    GpuReady,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertResidencyOutcome {
    ResidentHit,
    MaterializingHit,
    HostStagedHit,
    ColdMiss,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertPredictionReason {
    SessionHotset,
    GlobalHotset,
    CrossLayerTransition,
    WorkloadHotset,
    ColdMissBoost,
    Mixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertCacheAction {
    KeepResident,
    PrefetchToGpu,
    StageToHost,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExpertPrediction {
    pub expert: ExpertId,
    pub score: f32,
    pub confidence: f32,
    pub reason: ExpertPredictionReason,
    pub action: ExpertCacheAction,
}

#[derive(Debug, Clone, Copy)]
pub struct ExpertPredictContext<'a> {
    pub layer: usize,
    pub phase: ExpertAccessPhase,
    pub budget: usize,
    pub num_experts: usize,
    pub resident: &'a [usize],
    pub materializing: &'a [usize],
    pub host_staged: &'a [usize],
}

impl<'a> ExpertPredictContext<'a> {
    pub fn new(layer: usize, phase: ExpertAccessPhase, budget: usize, num_experts: usize) -> Self {
        Self {
            layer,
            phase,
            budget,
            num_experts,
            resident: &[],
            materializing: &[],
            host_staged: &[],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExpertAccessEvent {
    pub expert: ExpertId,
    pub route_weight: f32,
    pub phase: ExpertAccessPhase,
    pub outcome: ExpertResidencyOutcome,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExpertBatchExpertEvent {
    pub expert: ExpertId,
    pub columns: usize,
    pub total_route_weight: f32,
    pub outcome: ExpertResidencyOutcome,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExpertBatchAccessEvent {
    pub layer: usize,
    pub phase: ExpertAccessPhase,
    pub token_count: usize,
    pub experts: Vec<ExpertBatchExpertEvent>,
}

impl ExpertBatchAccessEvent {
    pub fn from_routes(
        layer: usize,
        phase: ExpertAccessPhase,
        token_count: usize,
        routes: &[ExpertRoute],
        streaming: &ExpertStreamingStep,
    ) -> Self {
        let routes_by_token = vec![routes.to_vec()];
        Self::from_routes_by_token(
            layer,
            phase,
            token_count.max(1),
            &routes_by_token,
            &[streaming.clone()],
        )
    }

    pub fn from_routes_by_token(
        layer: usize,
        phase: ExpertAccessPhase,
        token_count: usize,
        routes_by_token: &[Vec<ExpertRoute>],
        streaming_steps: &[ExpertStreamingStep],
    ) -> Self {
        let selected_loads = streaming_steps
            .iter()
            .flat_map(|step| step.loads.iter())
            .filter(|load| load.reason == ExpertLoadReason::Selected)
            .map(|load| load.expert)
            .collect::<BTreeSet<_>>();

        let mut aggregated = BTreeMap::<ExpertId, ExpertBatchExpertEvent>::new();
        for routes in routes_by_token {
            let mut seen_in_token = BTreeSet::<ExpertId>::new();
            for route in routes {
                let expert = ExpertId::new(layer, route.expert);
                let outcome = if selected_loads.contains(&expert) {
                    // The generic streaming step only knows that the selected
                    // expert was not planner-resident. CUDA materialization may
                    // later classify this more precisely as materializing or
                    // host-staged; for predictor scoring this is still the
                    // expensive feedback signal we need to boost future demand.
                    ExpertResidencyOutcome::ColdMiss
                } else {
                    ExpertResidencyOutcome::ResidentHit
                };
                let entry = aggregated.entry(expert).or_insert(ExpertBatchExpertEvent {
                    expert,
                    columns: 0,
                    total_route_weight: 0.0,
                    outcome,
                });
                if seen_in_token.insert(expert) {
                    entry.columns = entry.columns.saturating_add(1);
                }
                entry.total_route_weight += route.weight;
                if outcome == ExpertResidencyOutcome::ColdMiss {
                    entry.outcome = ExpertResidencyOutcome::ColdMiss;
                }
            }
        }

        Self {
            layer,
            phase,
            token_count: token_count.max(routes_by_token.len()).max(1),
            experts: aggregated.into_values().collect(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertPredictionStats {
    pub predict_calls: u64,
    pub predicted_experts: u64,
    pub observe_calls: u64,
    pub observed_experts: u64,
    pub cold_miss_observations: u64,
    pub transition_observations: u64,
}

pub trait ExpertHotsetPredictor {
    fn predict(&mut self, ctx: ExpertPredictContext<'_>) -> Vec<ExpertPrediction>;
    fn observe_batch(&mut self, event: ExpertBatchAccessEvent);
    fn stats(&self) -> ExpertPredictionStats;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreBasedExpertPredictorConfig {
    pub session_decay: f32,
    pub global_decay: f32,
    pub workload_decay: f32,
    pub cold_miss_decay: f32,
    pub transition_decay: f32,
    pub session_weight: f32,
    pub global_weight: f32,
    pub workload_weight: f32,
    pub cold_miss_weight: f32,
    pub transition_weight: f32,
    pub cold_miss_boost: f32,
    pub min_score: f32,
}

impl Default for ScoreBasedExpertPredictorConfig {
    fn default() -> Self {
        Self {
            session_decay: 0.92,
            global_decay: 0.995,
            workload_decay: 0.90,
            cold_miss_decay: 0.85,
            transition_decay: 0.96,
            session_weight: 1.0,
            global_weight: 0.35,
            workload_weight: 0.35,
            cold_miss_weight: 1.75,
            transition_weight: 1.50,
            cold_miss_boost: 2.0,
            min_score: 1.0e-6,
        }
    }
}

#[derive(Debug, Clone)]
struct LastLayerSelection {
    layer: usize,
    experts: Vec<(usize, f32)>,
}

#[derive(Debug, Clone)]
pub struct ScoreBasedExpertPredictor {
    num_layers: usize,
    num_experts: usize,
    config: ScoreBasedExpertPredictorConfig,
    session_scores: Vec<f32>,
    global_scores: Vec<f32>,
    workload_scores: Vec<f32>,
    cold_miss_scores: Vec<f32>,
    transition_scores: HashMap<(usize, usize, usize), f32>,
    last_layer: Option<LastLayerSelection>,
    stats: ExpertPredictionStats,
}

impl ScoreBasedExpertPredictor {
    pub fn new(num_layers: usize, num_experts: usize) -> Self {
        Self::with_config(
            num_layers,
            num_experts,
            ScoreBasedExpertPredictorConfig::default(),
        )
    }

    pub fn with_config(
        num_layers: usize,
        num_experts: usize,
        config: ScoreBasedExpertPredictorConfig,
    ) -> Self {
        let cells = num_layers.saturating_mul(num_experts);
        Self {
            num_layers,
            num_experts,
            config,
            session_scores: vec![0.0; cells],
            global_scores: vec![0.0; cells],
            workload_scores: vec![0.0; cells],
            cold_miss_scores: vec![0.0; cells],
            transition_scores: HashMap::new(),
            last_layer: None,
            stats: ExpertPredictionStats::default(),
        }
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Clear all session/workload/cold-miss state. Used by sequence reset.
    pub fn clear(&mut self) {
        self.session_scores.fill(0.0);
        self.global_scores.fill(0.0);
        self.workload_scores.fill(0.0);
        self.cold_miss_scores.fill(0.0);
        self.transition_scores.clear();
        self.last_layer = None;
    }

    pub fn score_for(&self, layer: usize, expert: usize) -> Option<f32> {
        self.index(layer, expert).map(|idx| {
            self.config.session_weight * self.session_scores[idx]
                + self.config.global_weight * self.global_scores[idx]
                + self.config.workload_weight * self.workload_scores[idx]
                + self.config.cold_miss_weight * self.cold_miss_scores[idx]
        })
    }

    fn index(&self, layer: usize, expert: usize) -> Option<usize> {
        (layer < self.num_layers && expert < self.num_experts)
            .then_some(layer * self.num_experts + expert)
    }

    fn decay_layer(scores: &mut [f32], num_experts: usize, layer: usize, decay: f32) {
        let start = layer.saturating_mul(num_experts);
        let end = start.saturating_add(num_experts).min(scores.len());
        for score in &mut scores[start..end] {
            *score *= decay;
        }
    }

    fn transition_score_for(&self, layer: usize, expert: usize) -> f32 {
        let Some(last) = &self.last_layer else {
            return 0.0;
        };
        if last.layer.saturating_add(1) != layer {
            return 0.0;
        }
        last.experts
            .iter()
            .map(|(prev_expert, prev_weight)| {
                self.transition_scores
                    .get(&(last.layer, *prev_expert, expert))
                    .copied()
                    .unwrap_or(0.0)
                    * prev_weight.max(0.0)
            })
            .sum()
    }

    fn prediction_action(
        resident: &[usize],
        materializing: &[usize],
        host_staged: &[usize],
        expert: usize,
    ) -> ExpertCacheAction {
        if resident.binary_search(&expert).is_ok() {
            ExpertCacheAction::KeepResident
        } else if host_staged.binary_search(&expert).is_ok()
            || materializing.binary_search(&expert).is_ok()
        {
            ExpertCacheAction::PrefetchToGpu
        } else {
            ExpertCacheAction::StageToHost
        }
    }

    fn reason(
        session: f32,
        global: f32,
        workload: f32,
        cold: f32,
        transition: f32,
    ) -> ExpertPredictionReason {
        let mut reasons = Vec::new();
        if session > 0.0 {
            reasons.push(ExpertPredictionReason::SessionHotset);
        }
        if global > 0.0 {
            reasons.push(ExpertPredictionReason::GlobalHotset);
        }
        if workload > 0.0 {
            reasons.push(ExpertPredictionReason::WorkloadHotset);
        }
        if cold > 0.0 {
            reasons.push(ExpertPredictionReason::ColdMissBoost);
        }
        if transition > 0.0 {
            reasons.push(ExpertPredictionReason::CrossLayerTransition);
        }
        if reasons.len() == 1 {
            reasons[0]
        } else {
            ExpertPredictionReason::Mixed
        }
    }
}

impl ExpertHotsetPredictor for ScoreBasedExpertPredictor {
    fn predict(&mut self, ctx: ExpertPredictContext<'_>) -> Vec<ExpertPrediction> {
        self.stats.predict_calls = self.stats.predict_calls.saturating_add(1);
        if ctx.budget == 0 || ctx.layer >= self.num_layers || self.num_experts == 0 {
            return Vec::new();
        }
        let num_experts = ctx.num_experts.min(self.num_experts);
        let mut resident = ctx.resident.to_vec();
        resident.sort_unstable();
        let mut materializing = ctx.materializing.to_vec();
        materializing.sort_unstable();
        let mut host_staged = ctx.host_staged.to_vec();
        host_staged.sort_unstable();

        let mut candidates = Vec::new();
        for expert in 0..num_experts {
            let idx = self.index(ctx.layer, expert).expect("bounded above");
            let session = self.config.session_weight * self.session_scores[idx];
            let global = self.config.global_weight * self.global_scores[idx];
            let workload = self.config.workload_weight * self.workload_scores[idx];
            let cold = self.config.cold_miss_weight * self.cold_miss_scores[idx];
            let transition =
                self.config.transition_weight * self.transition_score_for(ctx.layer, expert);
            let score = session + global + workload + cold + transition;
            if score <= self.config.min_score || !score.is_finite() {
                continue;
            }
            let action = Self::prediction_action(&resident, &materializing, &host_staged, expert);
            if action == ExpertCacheAction::KeepResident {
                continue;
            }
            candidates.push(ExpertPrediction {
                expert: ExpertId::new(ctx.layer, expert),
                score,
                confidence: score / (score + 1.0),
                reason: Self::reason(session, global, workload, cold, transition),
                action,
            });
        }
        candidates.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.expert.expert.cmp(&right.expert.expert))
        });
        candidates.truncate(ctx.budget);
        self.stats.predicted_experts = self
            .stats
            .predicted_experts
            .saturating_add(candidates.len() as u64);
        candidates
    }

    fn observe_batch(&mut self, event: ExpertBatchAccessEvent) {
        self.stats.observe_calls = self.stats.observe_calls.saturating_add(1);
        if event.layer >= self.num_layers {
            self.last_layer = None;
            return;
        }

        Self::decay_layer(
            &mut self.session_scores,
            self.num_experts,
            event.layer,
            self.config.session_decay,
        );
        Self::decay_layer(
            &mut self.global_scores,
            self.num_experts,
            event.layer,
            self.config.global_decay,
        );
        Self::decay_layer(
            &mut self.workload_scores,
            self.num_experts,
            event.layer,
            self.config.workload_decay,
        );
        Self::decay_layer(
            &mut self.cold_miss_scores,
            self.num_experts,
            event.layer,
            self.config.cold_miss_decay,
        );

        let mut observed = Vec::new();
        for expert_event in &event.experts {
            if expert_event.expert.layer != event.layer {
                continue;
            }
            let Some(idx) = self.index(event.layer, expert_event.expert.expert) else {
                continue;
            };
            let weight = expert_event.total_route_weight.max(0.0);
            let columns = expert_event.columns.max(1) as f32;
            self.session_scores[idx] += weight;
            self.global_scores[idx] += 0.25 * weight;
            self.workload_scores[idx] += columns;
            if matches!(expert_event.outcome, ExpertResidencyOutcome::ColdMiss) {
                self.cold_miss_scores[idx] += self.config.cold_miss_boost + weight;
                self.stats.cold_miss_observations =
                    self.stats.cold_miss_observations.saturating_add(1);
            }
            observed.push((expert_event.expert.expert, weight.max(1.0e-6)));
            self.stats.observed_experts = self.stats.observed_experts.saturating_add(1);
        }

        if let Some(last) = &self.last_layer {
            if last.layer.saturating_add(1) == event.layer {
                for (prev_expert, prev_weight) in &last.experts {
                    for (next_expert, next_weight) in &observed {
                        let key = (last.layer, *prev_expert, *next_expert);
                        let entry = self.transition_scores.entry(key).or_insert(0.0);
                        *entry = self.config.transition_decay * *entry + prev_weight * next_weight;
                        self.stats.transition_observations =
                            self.stats.transition_observations.saturating_add(1);
                    }
                }
            }
        }

        self.last_layer = Some(LastLayerSelection {
            layer: event.layer,
            experts: observed,
        });
    }

    fn stats(&self) -> ExpertPredictionStats {
        self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe::routing::ExpertRoute;
    use crate::moe::streaming::{
        ExpertLoadRequest, ExpertLoadSource, ExpertStreamingPlanner, ExpertStreamingPolicy,
    };
    use std::path::PathBuf;

    fn source() -> ExpertLoadSource {
        ExpertLoadSource::LocalShard {
            path: PathBuf::from("model.safetensors"),
            offset: 0,
            bytes: 10,
        }
    }

    #[test]
    fn score_predictor_boosts_observed_cold_misses() {
        let mut predictor = ScoreBasedExpertPredictor::new(2, 8);
        predictor.observe_batch(ExpertBatchAccessEvent {
            layer: 0,
            phase: ExpertAccessPhase::Decode,
            token_count: 1,
            experts: vec![ExpertBatchExpertEvent {
                expert: ExpertId::new(0, 3),
                columns: 1,
                total_route_weight: 0.7,
                outcome: ExpertResidencyOutcome::ColdMiss,
            }],
        });

        let predicted = predictor.predict(ExpertPredictContext::new(
            0,
            ExpertAccessPhase::Decode,
            2,
            8,
        ));
        assert_eq!(predicted[0].expert, ExpertId::new(0, 3));
        assert!(predicted[0].score > 0.7);
        assert!(matches!(
            predicted[0].reason,
            ExpertPredictionReason::Mixed | ExpertPredictionReason::ColdMissBoost
        ));
        assert_eq!(predictor.stats().cold_miss_observations, 1);
    }

    #[test]
    fn score_predictor_skips_already_resident_experts_for_prefetch() {
        let mut predictor = ScoreBasedExpertPredictor::new(1, 8);
        predictor.observe_batch(ExpertBatchAccessEvent {
            layer: 0,
            phase: ExpertAccessPhase::Decode,
            token_count: 1,
            experts: vec![
                ExpertBatchExpertEvent {
                    expert: ExpertId::new(0, 3),
                    columns: 1,
                    total_route_weight: 10.0,
                    outcome: ExpertResidencyOutcome::ResidentHit,
                },
                ExpertBatchExpertEvent {
                    expert: ExpertId::new(0, 4),
                    columns: 1,
                    total_route_weight: 1.0,
                    outcome: ExpertResidencyOutcome::ColdMiss,
                },
            ],
        });

        let predicted = predictor.predict(ExpertPredictContext {
            layer: 0,
            phase: ExpertAccessPhase::Decode,
            budget: 2,
            num_experts: 8,
            resident: &[3],
            materializing: &[],
            host_staged: &[],
        });
        assert_eq!(predicted.len(), 1);
        assert_eq!(predicted[0].expert, ExpertId::new(0, 4));
        assert_eq!(predicted[0].action, ExpertCacheAction::StageToHost);
    }

    #[test]
    fn score_predictor_learns_adjacent_layer_transitions() {
        let mut predictor = ScoreBasedExpertPredictor::new(4, 8);
        predictor.observe_batch(ExpertBatchAccessEvent {
            layer: 0,
            phase: ExpertAccessPhase::Decode,
            token_count: 1,
            experts: vec![ExpertBatchExpertEvent {
                expert: ExpertId::new(0, 2),
                columns: 1,
                total_route_weight: 1.0,
                outcome: ExpertResidencyOutcome::ResidentHit,
            }],
        });
        predictor.observe_batch(ExpertBatchAccessEvent {
            layer: 1,
            phase: ExpertAccessPhase::Decode,
            token_count: 1,
            experts: vec![ExpertBatchExpertEvent {
                expert: ExpertId::new(1, 5),
                columns: 1,
                total_route_weight: 1.0,
                outcome: ExpertResidencyOutcome::ResidentHit,
            }],
        });
        // Re-observe layer 0 as the current previous layer; predicting layer 1
        // should now use the learned 0:2 -> 1:5 transition.
        predictor.observe_batch(ExpertBatchAccessEvent {
            layer: 0,
            phase: ExpertAccessPhase::Decode,
            token_count: 1,
            experts: vec![ExpertBatchExpertEvent {
                expert: ExpertId::new(0, 2),
                columns: 1,
                total_route_weight: 1.0,
                outcome: ExpertResidencyOutcome::ResidentHit,
            }],
        });

        let predicted = predictor.predict(ExpertPredictContext::new(
            1,
            ExpertAccessPhase::Decode,
            1,
            8,
        ));
        assert_eq!(predicted[0].expert, ExpertId::new(1, 5));
        assert!(predictor.stats().transition_observations > 0);
    }

    #[test]
    fn access_event_from_routes_by_token_aggregates_batch_demand() {
        let source = source();
        let step0 = ExpertStreamingStep {
            layer: 0,
            selected: vec![ExpertId::new(0, 2), ExpertId::new(0, 3)],
            prefetched: Vec::new(),
            loads: vec![ExpertLoadRequest {
                expert: ExpertId::new(0, 2),
                load_source: source,
                reason: ExpertLoadReason::Selected,
            }],
            evictions: Vec::new(),
        };
        let step1 = ExpertStreamingStep {
            layer: 0,
            selected: vec![ExpertId::new(0, 3)],
            prefetched: Vec::new(),
            loads: Vec::new(),
            evictions: Vec::new(),
        };
        let event = ExpertBatchAccessEvent::from_routes_by_token(
            0,
            ExpertAccessPhase::Prefill,
            3,
            &[
                vec![
                    ExpertRoute {
                        expert: 2,
                        weight: 0.25,
                        score: 1.0,
                        selection_score: 1.0,
                    },
                    ExpertRoute {
                        expert: 3,
                        weight: 0.75,
                        score: 1.0,
                        selection_score: 1.0,
                    },
                ],
                vec![ExpertRoute {
                    expert: 3,
                    weight: 1.0,
                    score: 1.0,
                    selection_score: 1.0,
                }],
                vec![ExpertRoute {
                    expert: 2,
                    weight: 0.50,
                    score: 1.0,
                    selection_score: 1.0,
                }],
            ],
            &[step0, step1],
        );

        assert_eq!(event.token_count, 3);
        assert_eq!(event.experts.len(), 2);
        let by_expert = event
            .experts
            .iter()
            .map(|event| (event.expert.expert, event))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(by_expert[&2].columns, 2);
        assert_eq!(by_expert[&2].total_route_weight, 0.75);
        assert_eq!(by_expert[&2].outcome, ExpertResidencyOutcome::ColdMiss);
        assert_eq!(by_expert[&3].columns, 2);
        assert_eq!(by_expert[&3].total_route_weight, 1.75);
        assert_eq!(by_expert[&3].outcome, ExpertResidencyOutcome::ResidentHit);
    }

    #[test]
    fn access_event_from_routes_marks_selected_loads_as_misses() {
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy {
            gpu_slots_per_layer: 2,
            prefetch_per_layer: 0,
            preserve_artifact_quantization: true,
            allow_cpu_staging: false,
            allow_remote_sources: false,
        });
        for expert in 0..4 {
            planner.register_load_source(ExpertId::new(0, expert), source());
        }
        let step = planner.plan_layer_step(0, &[2], &[]).unwrap();
        let event = ExpertBatchAccessEvent::from_routes(
            0,
            ExpertAccessPhase::Decode,
            1,
            &[ExpertRoute {
                expert: 2,
                weight: 0.5,
                score: 1.0,
                selection_score: 1.0,
            }],
            &step,
        );
        assert_eq!(event.experts.len(), 1);
        assert_eq!(event.experts[0].outcome, ExpertResidencyOutcome::ColdMiss);
    }
}
