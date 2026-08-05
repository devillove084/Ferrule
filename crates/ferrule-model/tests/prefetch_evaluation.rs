//! Deterministic simulation evaluating whether score-based expert prefetch
//! actually reduces demand-miss (uncovered expert-I/O wait) events.
//!
//! The experiment replays one synthetic decode trace through four cache
//! policies:
//!
//! - `demand-only`: per-layer LRU, loads only routed experts (today's
//!   production behavior);
//! - `prefetch`: the same LRU plus the production
//!   [`ScoreBasedExpertPredictor`]. Predictions arrive after a fixed I/O
//!   latency and may occupy up to `PREFETCH_ALLOWANCE` slots per layer —
//!   the demand-reserve semantics `ResourceBroker` actually enforces
//!   (prefetch share is capped, demand keeps its reserve);
//! - `oracle`: perfect foresight prefetch under the same allowance, bounding
//!   the achievable benefit.
//!
//! The trace combines a global Zipf popularity ranking (the contested
//! mid-rank band where a frequency-aware predictor can beat LRU), a session
//! hot subset, and a learnable cross-layer transition mapping. `rho`/`tau`
//! sweep the correlation structure so the result states the decision
//! boundary: which routing structure makes prefetch pay off, and whether it
//! can ever hurt.
//!
//! # Findings (2026-07, seeds fixed)
//!
//! - Oracle prefetch under the same slot allowance cuts misses by ~95%, so
//!   the mechanism has real headroom.
//! - The production frequency-based predictor is *net negative* (~+30%
//!   misses) under slot competition: once the hot working set is resident,
//!   frequency scores can only nominate second-tier guesses that evict
//!   useful mid-rank entries. Disabling cold-miss chasing does not change
//!   this materially.
//! - On memoryless (iid) traffic the predictor stays silent and harmless.
//!
//! Conclusion: do not wire the current predictor into prefetch. Useful
//! prefetch requires a future-aware signal (hidden-state lookahead, routing
//! hash, or a learned predictor), not popularity statistics. Keep
//! `ResourceDemand::ModelWarmup`/demand-reserve in the runtime as the correct
//! carrier for that future signal.

use std::collections::{BTreeMap, VecDeque};

use ferrule_model::moe::{
    ExpertAccessPhase, ExpertBatchAccessEvent, ExpertBatchExpertEvent, ExpertHotsetPredictor,
    ExpertId, ExpertPredictContext, ExpertResidency, ExpertResidencyOutcome,
    ScoreBasedExpertPredictor,
};

const LAYERS: usize = 4;
const EXPERTS: usize = 256;
const TOP_K: usize = 8;
const STEPS: usize = 2_000;
const SLOTS_PER_LAYER: usize = 48;
const SESSION_HOT: usize = 16;
const PREFETCH_BUDGET: usize = 8;
const PREFETCH_ALLOWANCE: usize = 16;
const PREFETCH_LATENCY: u64 = 3;

/// SplitMix64: tiny deterministic PRNG so the simulation needs no `rand` dep.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }

    fn chance(&mut self, numerator: u64, denominator: u64) -> bool {
        self.next() % denominator < numerator
    }

    /// Zipf sample over `n` ranks, weight ∝ 1/(rank + 1).
    fn zipf(&mut self, n: usize) -> usize {
        let mut rank = 0usize;
        while rank + 1 < n && self.chance(2, 3) {
            rank += 1;
        }
        rank
    }
}

#[derive(Debug, Clone, Copy)]
struct Scenario {
    name: &'static str,
    seed: u64,
    /// Probability that one routed slot comes from the session hot subset.
    rho: u64,
    /// Probability that one routed slot follows the cross-layer transition.
    tau: u64,
}

/// Fixed bijection used as the learnable cross-layer transition mapping.
fn transition_partner(expert: usize) -> usize {
    (expert * 31 + 7) % EXPERTS
}

/// Generates the routed expert set for every (step, layer) of the trace.
///
/// Base traffic is global-Zipf over a fixed per-layer ranking: the head is
/// captured by any LRU, the mid-rank band (ranks ~32..96) is where a
/// frequency-aware predictor can outperform recency-only eviction.
fn generate_trace(scenario: Scenario) -> Vec<Vec<Vec<usize>>> {
    let mut rng = Rng(scenario.seed);
    let rankings: Vec<Vec<usize>> = (0..LAYERS)
        .map(|_| {
            let mut ranking: Vec<usize> = (0..EXPERTS).collect();
            for index in (1..ranking.len()).rev() {
                ranking.swap(index, rng.below(index + 1));
            }
            ranking
        })
        .collect();
    let session_hot: Vec<Vec<usize>> = rankings
        .iter()
        .map(|ranking| ranking[32..32 + SESSION_HOT].to_vec())
        .collect();

    let mut trace: Vec<Vec<Vec<usize>>> = (0..STEPS)
        .map(|_| (0..LAYERS).map(|_| Vec::with_capacity(TOP_K)).collect())
        .collect();
    for layers in trace.iter_mut() {
        for layer in 0..LAYERS {
            let (previous, current) = layers.split_at_mut(layer);
            let routed = &mut current[0];
            while routed.len() < TOP_K {
                let expert = if layer > 0 && scenario.tau > 0 && rng.chance(scenario.tau, 100) {
                    let prev_layer = &previous[layer - 1];
                    transition_partner(prev_layer[rng.below(prev_layer.len().max(1))])
                } else if rng.chance(scenario.rho, 100) {
                    session_hot[layer][rng.zipf(SESSION_HOT)]
                } else {
                    rankings[layer][rng.zipf(EXPERTS)]
                };
                if !routed.contains(&expert) {
                    routed.push(expert);
                }
            }
        }
    }
    trace
}

#[derive(Debug, Clone, Copy)]
struct CacheEntry {
    last_used: u64,
    /// Counts against the prefetch allowance (vs demand-owned slot).
    prefetch: bool,
}

#[derive(Debug, Default)]
struct LayerCache {
    // BTreeMap: deterministic iteration order keeps LRU tie-breaking stable.
    entries: BTreeMap<usize, CacheEntry>,
    prefetch_occupied: usize,
}

impl LayerCache {
    fn evict_lru(&mut self) {
        if let Some(victim) = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(expert, _)| *expert)
            && self
                .entries
                .remove(&victim)
                .is_some_and(|entry| entry.prefetch)
        {
            self.prefetch_occupied -= 1;
        }
    }

    fn access(&mut self, expert: usize, now: u64, metrics: &mut Metrics) -> bool {
        if let Some(entry) = self.entries.get_mut(&expert) {
            entry.last_used = now;
            if entry.prefetch {
                entry.prefetch = false;
                self.prefetch_occupied -= 1;
                metrics.prefetch_used += 1;
            }
            return true;
        }
        if self.entries.len() >= SLOTS_PER_LAYER {
            self.evict_lru();
        }
        self.entries.insert(
            expert,
            CacheEntry {
                last_used: now,
                prefetch: false,
            },
        );
        false
    }

    /// Inserts one predicted expert while the prefetch share of this layer is
    /// below its allowance. Demand-owned slots are never taken.
    fn prefetch_insert(&mut self, expert: usize, now: u64) -> bool {
        if self.entries.contains_key(&expert) || self.prefetch_occupied >= PREFETCH_ALLOWANCE {
            return false;
        }
        if self.entries.len() >= SLOTS_PER_LAYER {
            self.evict_lru();
            if self.entries.len() >= SLOTS_PER_LAYER {
                return false;
            }
        }
        self.entries.insert(
            expert,
            CacheEntry {
                last_used: now,
                prefetch: true,
            },
        );
        self.prefetch_occupied += 1;
        true
    }

    fn residency(&self) -> Vec<ExpertResidency> {
        let mut residency = vec![ExpertResidency::Cold; EXPERTS];
        for expert in self.entries.keys() {
            residency[*expert] = ExpertResidency::GpuReady;
        }
        residency
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct Metrics {
    misses: u64,
    prefetch_delivered: u64,
    prefetch_used: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Policy {
    DemandOnly,
    /// Production predictor with default config (cold-miss chasing included).
    Prefetch,
    /// Predictor with cold-miss chasing disabled: only sustained frequency
    /// and transition signals may occupy the prefetch allowance.
    PrefetchTuned,
    Oracle,
}

fn simulate(trace: &[Vec<Vec<usize>>], policy: Policy) -> Metrics {
    let mut metrics = Metrics::default();
    let mut caches: Vec<LayerCache> = (0..LAYERS).map(|_| LayerCache::default()).collect();
    let mut predictor = match policy {
        Policy::PrefetchTuned => ScoreBasedExpertPredictor::with_config(
            LAYERS,
            EXPERTS,
            ferrule_model::moe::ScoreBasedExpertPredictorConfig {
                cold_miss_weight: 0.0,
                cold_miss_boost: 0.0,
                ..Default::default()
            },
        ),
        _ => ScoreBasedExpertPredictor::new(LAYERS, EXPERTS),
    };
    let mut pending: VecDeque<(u64, usize, usize)> = VecDeque::new();

    for (step, layers) in trace.iter().enumerate() {
        let now = step as u64;
        // 1. Deliver prefetches whose I/O latency has elapsed.
        while pending.front().is_some_and(|(ready, _, _)| *ready <= now) {
            let (_, layer, expert) = pending.pop_front().expect("checked above");
            if caches[layer].prefetch_insert(expert, now) {
                metrics.prefetch_delivered += 1;
            }
        }
        for (layer, routed) in layers.iter().enumerate() {
            // 2. Ask the predictor (or oracle) for this layer's prefetch set.
            match policy {
                Policy::DemandOnly => {}
                Policy::Prefetch | Policy::PrefetchTuned => {
                    let residency = caches[layer].residency();
                    let mut ctx = ExpertPredictContext::new(
                        layer,
                        ExpertAccessPhase::Decode,
                        PREFETCH_BUDGET,
                        EXPERTS,
                    );
                    ctx.residency = &residency;
                    for prediction in predictor.predict(ctx) {
                        let already_pending = pending.iter().any(|(_, pending_layer, expert)| {
                            *pending_layer == layer && *expert == prediction.expert.expert
                        });
                        if !already_pending {
                            pending.push_back((
                                now + PREFETCH_LATENCY,
                                layer,
                                prediction.expert.expert,
                            ));
                        }
                    }
                }
                Policy::Oracle => {
                    let future = (now + PREFETCH_LATENCY) as usize;
                    if future < trace.len() {
                        for &expert in &trace[future][layer] {
                            pending.push_back((now + PREFETCH_LATENCY, layer, expert));
                        }
                    }
                }
            }
            // 3. Demand access: every miss is one uncovered expert-I/O wait.
            let mut experts = Vec::with_capacity(routed.len());
            for &expert in routed {
                let hit = caches[layer].access(expert, now, &mut metrics);
                if !hit {
                    metrics.misses += 1;
                }
                experts.push(ExpertBatchExpertEvent {
                    expert: ExpertId::new(layer, expert),
                    columns: 1,
                    total_route_weight: 1.0,
                    outcome: if hit {
                        ExpertResidencyOutcome::ResidentHit
                    } else {
                        ExpertResidencyOutcome::ColdMiss
                    },
                });
            }
            // 4. Feed the predictor exactly like the production runner does.
            if matches!(policy, Policy::Prefetch | Policy::PrefetchTuned) {
                predictor.observe_batch(ExpertBatchAccessEvent {
                    layer,
                    phase: ExpertAccessPhase::Decode,
                    token_count: 1,
                    experts,
                });
            }
        }
    }
    metrics
}

fn report(scenario: Scenario) -> (Metrics, Metrics, Metrics, Metrics) {
    let trace = generate_trace(scenario);
    let demand = simulate(&trace, Policy::DemandOnly);
    let prefetch = simulate(&trace, Policy::Prefetch);
    let tuned = simulate(&trace, Policy::PrefetchTuned);
    let oracle = simulate(&trace, Policy::Oracle);
    eprintln!(
        "scenario={} rho={} tau={} | demand={} | prefetch={} (-{:.1}%, used {}/{}) | tuned={} (-{:.1}%, used {}/{}) | oracle={} (-{:.1}%)",
        scenario.name,
        scenario.rho,
        scenario.tau,
        demand.misses,
        prefetch.misses,
        100.0 * (1.0 - prefetch.misses as f64 / demand.misses.max(1) as f64),
        prefetch.prefetch_used,
        prefetch.prefetch_delivered,
        tuned.misses,
        100.0 * (1.0 - tuned.misses as f64 / demand.misses.max(1) as f64),
        tuned.prefetch_used,
        tuned.prefetch_delivered,
        oracle.misses,
        100.0 * (1.0 - oracle.misses as f64 / demand.misses.max(1) as f64),
    );
    (demand, prefetch, tuned, oracle)
}

#[test]
fn oracle_headroom_exists_but_current_predictor_is_net_negative() {
    let (demand, prefetch, tuned, oracle) = report(Scenario {
        name: "correlated",
        seed: 7,
        rho: 85,
        tau: 60,
    });
    assert!(
        oracle.misses * 100 <= demand.misses * 50,
        "oracle must cut misses by >=50% under the same allowance, proving mechanism headroom (demand={}, oracle={})",
        demand.misses,
        oracle.misses
    );
    assert!(
        prefetch.misses >= demand.misses,
        "decision record: the frequency-based predictor must not be wired as-is; it is net negative under slot competition (demand={}, prefetch={})",
        demand.misses,
        prefetch.misses
    );
    assert!(
        tuned.misses * 100 <= prefetch.misses * 110,
        "dropping cold-miss chasing is not materially different and does not rescue the predictor (default={}, tuned={})",
        prefetch.misses,
        tuned.misses
    );
}

#[test]
fn prefetch_never_hurts_when_routing_is_pure_noise() {
    let (demand, prefetch, tuned, _) = report(Scenario {
        name: "iid",
        seed: 11,
        rho: 0,
        tau: 0,
    });
    for (label, metrics) in [("default", prefetch), ("tuned", tuned)] {
        assert!(
            metrics.misses * 100 <= demand.misses * 102,
            "capped prefetch share: {label} policy must not add misses on iid routing (demand={}, {label}={})",
            demand.misses,
            metrics.misses
        );
    }
}

#[test]
fn prefetch_decision_boundary_sweep() {
    // Not an assertion test: prints the usefulness boundary for the record.
    for (rho, tau) in [(0u64, 0u64), (30, 0), (60, 30), (85, 60), (95, 80)] {
        let (demand, _, tuned, _) = report(Scenario {
            name: "sweep",
            seed: 23,
            rho,
            tau,
        });
        eprintln!(
            "  boundary rho={rho} tau={tau}: tuned prefetch keeps {:.1}% of demand misses",
            100.0 * tuned.misses as f64 / demand.misses.max(1) as f64
        );
    }
}
