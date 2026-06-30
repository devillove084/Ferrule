//! Expert residency policy — manage which experts are GPU-resident, CPU-resident,
//! or offloaded to NVMe/qcache. Works with expert activation telemetry to decide
//! hot/cold placement.

use std::collections::HashMap;

/// Prefetch policy: predict next-token experts based on router history.
pub struct PrefetchPolicy {
    /// Window of recent router decisions per layer.
    pub history_window: usize,
    /// Number of experts to prefetch per layer.
    pub prefetch_count: usize,
}

impl Default for PrefetchPolicy {
    fn default() -> Self {
        Self {
            history_window: 4,
            prefetch_count: 2,
        }
    }
}

impl PrefetchPolicy {
    /// Predict which experts are likely needed next, based on recent history.
    /// Simple strategy: prefetch experts that appeared in the last N tokens.
    pub fn predict(&self, history: &[Vec<usize>]) -> Vec<usize> {
        let mut counts = std::collections::HashMap::new();
        for token_experts in history.iter().rev().take(self.history_window) {
            for &eid in token_experts {
                *counts.entry(eid).or_insert(0) += 1;
            }
        }
        let mut ranked: Vec<_> = counts.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        ranked.truncate(self.prefetch_count);
        ranked.into_iter().map(|(eid, _)| eid).collect()
    }
}

/// Where an expert's weights currently reside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertLocation {
    /// Weights uploaded to GPU VRAM, ready for inference.
    Gpu,
    /// Weights in CPU RAM, can be uploaded on demand.
    Cpu,
    /// Weights on NVMe/qcache file, slowest to access.
    Disk,
    /// Currently being transferred to GPU.
    Loading,
}

impl ExpertLocation {
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Gpu)
    }

    pub fn access_cost_ns(&self) -> u64 {
        match self {
            Self::Gpu => 1,
            Self::Cpu => 10_000,
            Self::Disk => 10_000_000,
            Self::Loading => u64::MAX,
        }
    }
}

/// Which expert to keep resident, based on activation patterns.
#[derive(Debug, Clone)]
pub enum ResidencyStrategy {
    /// Keep top-N most-activated experts per layer on GPU.
    TopN(usize),
    /// Keep experts activated above a threshold fraction on GPU.
    Threshold(f64),
    /// Fixed assignment: explicit expert→location map.
    Fixed(HashMap<usize, ExpertLocation>),
    /// All experts on GPU (default, no offload).
    AllGpu,
}

/// Per-layer residency state: maps expert index → current location.
#[derive(Debug, Clone)]
pub struct LayerResidency {
    pub layer_idx: usize,
    /// Current location of each expert.
    pub locations: Vec<ExpertLocation>,
    /// Activation count per expert (from telemetry).
    pub hit_counts: Vec<usize>,
    /// Total tokens processed for this layer.
    pub total_tokens: usize,
}

impl LayerResidency {
    pub fn new(layer_idx: usize, num_experts: usize) -> Self {
        Self {
            layer_idx,
            locations: vec![ExpertLocation::Gpu; num_experts],
            hit_counts: vec![0; num_experts],
            total_tokens: 0,
        }
    }

    /// Record one activation of an expert. Called from the forward pass.
    pub fn record_hit(&mut self, expert_idx: usize) {
        if expert_idx < self.hit_counts.len() {
            self.hit_counts[expert_idx] = self.hit_counts[expert_idx].saturating_add(1);
        }
        self.total_tokens = self.total_tokens.saturating_add(1);
    }

    /// Get experts sorted by activation frequency (most → least).
    pub fn ranked(&self) -> Vec<(usize, usize)> {
        let mut pairs: Vec<_> = self.hit_counts.iter().copied().enumerate().collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs
    }

    /// Get the activation fraction for each expert [0.0, 1.0].
    pub fn fractions(&self) -> Vec<f64> {
        let denom = self.total_tokens.max(1) as f64;
        self.hit_counts.iter().map(|&c| c as f64 / denom).collect()
    }

    /// Apply a residency strategy: decide which experts should be GPU-resident.
    /// Returns (to_gpu, to_cpu, to_disk) — lists of expert indices to move.
    pub fn apply_strategy(
        &mut self,
        strategy: &ResidencyStrategy,
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let mut to_gpu = Vec::new();
        let mut to_cpu = Vec::new();
        let mut to_disk = Vec::new();

        match strategy {
            ResidencyStrategy::AllGpu => {
                for (i, loc) in self.locations.iter_mut().enumerate() {
                    if *loc != ExpertLocation::Gpu {
                        to_gpu.push(i);
                        *loc = ExpertLocation::Gpu;
                    }
                }
            }
            ResidencyStrategy::TopN(n) => {
                let ranked = self.ranked();
                for (i, &(eid, _)) in ranked.iter().enumerate() {
                    let target = if i < *n {
                        ExpertLocation::Gpu
                    } else {
                        ExpertLocation::Cpu
                    };
                    if self.locations[eid] != target {
                        match target {
                            ExpertLocation::Gpu => to_gpu.push(eid),
                            ExpertLocation::Cpu => to_cpu.push(eid),
                            _ => {}
                        }
                        self.locations[eid] = target;
                    }
                }
            }
            ResidencyStrategy::Threshold(t) => {
                let fractions = self.fractions();
                for (eid, &frac) in fractions.iter().enumerate() {
                    let target = if frac >= *t {
                        ExpertLocation::Gpu
                    } else {
                        ExpertLocation::Cpu
                    };
                    if self.locations[eid] != target {
                        match target {
                            ExpertLocation::Gpu => to_gpu.push(eid),
                            ExpertLocation::Cpu => to_cpu.push(eid),
                            _ => {}
                        }
                        self.locations[eid] = target;
                    }
                }
            }
            ResidencyStrategy::Fixed(map) => {
                for (&eid, &target) in map {
                    if eid < self.locations.len() && self.locations[eid] != target {
                        match target {
                            ExpertLocation::Gpu => to_gpu.push(eid),
                            ExpertLocation::Cpu => to_cpu.push(eid),
                            ExpertLocation::Disk => to_disk.push(eid),
                            _ => {}
                        }
                        self.locations[eid] = target;
                    }
                }
            }
        }

        (to_gpu, to_cpu, to_disk)
    }

    /// Number of GPU-resident experts.
    pub fn gpu_count(&self) -> usize {
        self.locations
            .iter()
            .filter(|l| matches!(l, ExpertLocation::Gpu))
            .count()
    }
}

/// Global residency manager across all layers.
#[derive(Debug, Default)]
pub struct ResidencyManager {
    pub layers: Vec<LayerResidency>,
}

impl ResidencyManager {
    pub fn new(num_layers: usize, num_experts: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|li| LayerResidency::new(li, num_experts))
                .collect(),
        }
    }

    pub fn record_hit(&mut self, layer: usize, expert: usize) {
        if let Some(lr) = self.layers.get_mut(layer) {
            lr.record_hit(expert);
        }
    }

    /// Apply strategy to all layers. Returns consolidated move lists.
    pub fn apply_all(&mut self, strategy: &ResidencyStrategy) -> ResidencyChangeSet {
        let mut cs = ResidencyChangeSet::default();
        for lr in &mut self.layers {
            let (tg, tc, td) = lr.apply_strategy(strategy);
            cs.to_gpu.extend(tg.iter().map(|&e| (lr.layer_idx, e)));
            cs.to_cpu.extend(tc.iter().map(|&e| (lr.layer_idx, e)));
            cs.to_disk.extend(td.iter().map(|&e| (lr.layer_idx, e)));
        }
        cs
    }

    pub fn gpu_resident_count(&self) -> usize {
        self.layers.iter().map(|lr| lr.gpu_count()).sum()
    }

    pub fn total_experts(&self) -> usize {
        self.layers.iter().map(|lr| lr.locations.len()).sum()
    }
}

/// A set of expert location changes produced by applying a residency strategy.
#[derive(Debug, Default, Clone)]
pub struct ResidencyChangeSet {
    /// (layer, expert) to move to GPU.
    pub to_gpu: Vec<(usize, usize)>,
    /// (layer, expert) to move to CPU.
    pub to_cpu: Vec<(usize, usize)>,
    /// (layer, expert) to move to disk.
    pub to_disk: Vec<(usize, usize)>,
}

impl ResidencyChangeSet {
    pub fn total_changes(&self) -> usize {
        self.to_gpu.len() + self.to_cpu.len() + self.to_disk.len()
    }
}

/// Expert batching strategy (P12.4).
///
/// When multiple sequences in a batch select the same expert, their GEMV
/// operations can be batched into a single GEMM for better GPU utilization.
/// This requires:
///
/// 1. Multi-sequence decode (from P8.3 continuous batching)
/// 2. Group sequences by selected experts per layer
/// 3. Launch batched GEMM instead of per-sequence GEMV
///
/// Current status: single-sequence only, no batching.
pub enum ExpertBatchingStrategy {
    /// One GEMV per sequence per expert (current).
    PerSequence,
    /// Batch same-expert GEMVs into GEMM when multiple sequences share.
    BatchSameExpert,
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_gpu_by_default() {
        let lr = LayerResidency::new(0, 8);
        assert_eq!(lr.gpu_count(), 8);
        for loc in &lr.locations {
            assert_eq!(*loc, ExpertLocation::Gpu);
        }
    }

    #[test]
    fn record_and_rank() {
        let mut lr = LayerResidency::new(0, 4);
        lr.record_hit(0);
        lr.record_hit(0);
        lr.record_hit(2);
        lr.record_hit(3);
        lr.record_hit(3);
        lr.record_hit(3);

        let ranked = lr.ranked();
        assert_eq!(ranked[0], (3, 3)); // expert 3: 3 hits
        assert_eq!(ranked[1], (0, 2)); // expert 0: 2 hits
        assert_eq!(ranked[2], (2, 1)); // expert 2: 1 hit
        assert_eq!(ranked[3], (1, 0)); // expert 1: 0 hits
    }

    #[test]
    fn top_n_strategy() {
        let mut lr = LayerResidency::new(0, 4);
        for _ in 0..10 {
            lr.record_hit(0);
        }
        for _ in 0..5 {
            lr.record_hit(1);
        }
        // Experts 2,3 have 0 hits

        let (to_gpu, to_cpu, _) = lr.apply_strategy(&ResidencyStrategy::TopN(2));
        assert_eq!(to_gpu.len(), 0); // top 2 already on GPU
        assert_eq!(to_cpu.len(), 2); // experts 2,3 evicted
        assert_eq!(lr.gpu_count(), 2);
    }

    #[test]
    fn threshold_strategy() {
        let mut lr = LayerResidency::new(0, 4);
        // 100 tokens, expert 0: 80 hits (80%), expert 1: 20 hits (20%)
        for _ in 0..80 {
            lr.record_hit(0);
        }
        for _ in 0..20 {
            lr.record_hit(1);
        }

        let (to_gpu, to_cpu, _) = lr.apply_strategy(&ResidencyStrategy::Threshold(0.5));
        assert_eq!(to_gpu.len(), 0); // expert 0 already GPU
        assert_eq!(to_cpu.len(), 3); // experts 1,2,3 below threshold
        assert!(lr.locations[0] == ExpertLocation::Gpu);
    }

    #[test]
    fn fixed_strategy() {
        let mut lr = LayerResidency::new(0, 3);
        let mut map = HashMap::new();
        map.insert(0, ExpertLocation::Gpu);
        map.insert(1, ExpertLocation::Cpu);
        map.insert(2, ExpertLocation::Disk);

        let (_to_gpu, to_cpu, to_disk) = lr.apply_strategy(&ResidencyStrategy::Fixed(map));
        assert_eq!(to_cpu.len(), 1); // expert 1 moved GPU→CPU
        assert_eq!(to_disk.len(), 1); // expert 2 moved GPU→Disk
        assert_eq!(lr.locations[0], ExpertLocation::Gpu);
        assert_eq!(lr.locations[1], ExpertLocation::Cpu);
        assert_eq!(lr.locations[2], ExpertLocation::Disk);
    }

    #[test]
    fn manager_aggregates() {
        let mut mgr = ResidencyManager::new(2, 4);
        mgr.record_hit(0, 0);
        mgr.record_hit(0, 0);
        mgr.record_hit(1, 3);

        assert_eq!(mgr.gpu_resident_count(), 8); // all 2×4 = 8 on GPU
        assert_eq!(mgr.total_experts(), 8);

        let cs = mgr.apply_all(&ResidencyStrategy::TopN(2));
        assert!(cs.total_changes() >= 4); // 2 layers × 2 evicts each
        assert_eq!(mgr.gpu_resident_count(), 4); // 2 layers × 2 experts each
    }

    #[test]
    fn location_access_cost_ordering() {
        assert!(ExpertLocation::Gpu.access_cost_ns() < ExpertLocation::Cpu.access_cost_ns());
        assert!(ExpertLocation::Cpu.access_cost_ns() < ExpertLocation::Disk.access_cost_ns());
        assert!(ExpertLocation::Disk.access_cost_ns() < ExpertLocation::Loading.access_cost_ns());
    }

    #[test]
    fn prefetch_returns_most_frequent() {
        let policy = PrefetchPolicy::default();
        // History: expert 3 appears 3 times, expert 1 appears 2 times, expert 0 appears once
        let history = vec![vec![0, 3], vec![1, 3], vec![1, 3], vec![2, 4]];
        let predicted = policy.predict(&history);
        assert_eq!(predicted.len(), 2);
        assert_eq!(predicted[0], 3); // most frequent
        assert_eq!(predicted[1], 1); // second most frequent
    }

    #[test]
    fn prefetch_respects_window() {
        let policy = PrefetchPolicy {
            history_window: 1,
            prefetch_count: 2,
        };
        // Only the last token's experts should be considered
        let history = vec![
            vec![5, 6],
            vec![7, 8],
            vec![8, 9], // last
        ];
        let predicted = policy.predict(&history);
        assert_eq!(predicted.len(), 2);
        assert!(predicted.contains(&8));
        assert!(predicted.contains(&9));
    }

    #[test]
    fn prefetch_empty_history() {
        let policy = PrefetchPolicy::default();
        let history: Vec<Vec<usize>> = vec![];
        let predicted = policy.predict(&history);
        assert!(predicted.is_empty());
    }

    #[test]
    fn prefetch_truncates_to_count() {
        let policy = PrefetchPolicy {
            history_window: 4,
            prefetch_count: 1,
        };
        let history = vec![vec![3, 5], vec![1, 7], vec![3, 9], vec![1, 11]];
        let predicted = policy.predict(&history);
        assert_eq!(predicted.len(), 1);
        // Either 1 or 3 is the most frequent; both appear twice, so accept either
        assert!(predicted[0] == 3 || predicted[0] == 1);
    }
}
