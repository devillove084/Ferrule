//! Per-layer expert activation telemetry (P12.1).
//!
//! Tracks expert selection counts across layers during inference,
//! enabling load-balancing analysis, utilization reports, and
//! expert popularity rankings.

use std::collections::BTreeMap;

/// Accumulates per-layer expert selection counts over many forward steps.
#[derive(Debug, Clone, Default)]
pub struct ExpertTelemetry {
    /// Per-layer: expert_id → selection count
    pub selections: BTreeMap<usize, BTreeMap<usize, u64>>,
    /// Total forward steps recorded (each step may record multiple layers).
    pub total_steps: u64,
}

impl ExpertTelemetry {
    /// Create an empty telemetry accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one layer's expert selections for a single forward step.
    ///
    /// `layer` is the 0-based layer index. `selected_experts` are the
    /// expert indices chosen by the router for this layer.
    pub fn record(&mut self, layer: usize, selected_experts: &[usize]) {
        let layer_counts = self.selections.entry(layer).or_default();
        for &expert in selected_experts {
            *layer_counts.entry(expert).or_default() += 1;
        }
        self.total_steps += 1;
    }

    /// Return the top-`k` most-selected experts for `layer`, sorted
    /// descending by selection count.
    pub fn top_experts(&self, layer: usize, k: usize) -> Vec<(usize, u64)> {
        let mut entries: Vec<_> = self
            .selections
            .get(&layer)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();
        entries.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        entries.truncate(k);
        entries
    }

    /// Fraction of `total_experts` that were selected at least once for
    /// `layer`. Returns 0.0 when no data has been recorded for the layer.
    pub fn utilization(&self, layer: usize, total_experts: usize) -> f64 {
        let active = self.selections.get(&layer).map(|m| m.len()).unwrap_or(0);
        active as f64 / total_experts.max(1) as f64
    }

    /// Total number of layers tracked.
    pub fn num_layers(&self) -> usize {
        self.selections.len()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_empty() {
        let t = ExpertTelemetry::new();
        assert_eq!(t.total_steps, 0);
        assert!(t.selections.is_empty());
    }

    #[test]
    fn record_single_layer_single_expert() {
        let mut t = ExpertTelemetry::new();
        t.record(0, &[3]);
        assert_eq!(t.total_steps, 1);
        assert_eq!(t.selections[&0][&3], 1);
    }

    #[test]
    fn record_multiple_experts_per_layer() {
        let mut t = ExpertTelemetry::new();
        t.record(0, &[0, 7, 3]);
        t.record(0, &[7, 9]);
        assert_eq!(t.selections[&0][&0], 1);
        assert_eq!(t.selections[&0][&7], 2);
        assert_eq!(t.selections[&0][&3], 1);
        assert_eq!(t.selections[&0][&9], 1);
        assert_eq!(t.total_steps, 2);
    }

    #[test]
    fn record_multi_layer() {
        let mut t = ExpertTelemetry::new();
        t.record(0, &[1, 2]);
        t.record(1, &[3, 4]);
        t.record(2, &[5]);
        assert_eq!(t.num_layers(), 3);
        assert_eq!(t.total_steps, 3);
    }

    #[test]
    fn top_experts_returns_correct_order() {
        let mut t = ExpertTelemetry::new();
        for _ in 0..10 {
            t.record(0, &[0]);
        }
        for _ in 0..5 {
            t.record(0, &[1]);
        }
        for _ in 0..15 {
            t.record(0, &[2]);
        }
        let top = t.top_experts(0, 2);
        assert_eq!(top, vec![(2, 15), (0, 10)]);

        let all = t.top_experts(0, 10);
        assert_eq!(all, vec![(2, 15), (0, 10), (1, 5)]);
    }

    #[test]
    fn top_experts_missing_layer_is_empty() {
        let t = ExpertTelemetry::new();
        assert!(t.top_experts(99, 5).is_empty());
    }

    #[test]
    fn utilization_full() {
        let mut t = ExpertTelemetry::new();
        t.record(0, &[0]);
        t.record(0, &[1]);
        t.record(0, &[2]);
        t.record(0, &[3]);
        assert!((t.utilization(0, 4) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn utilization_partial() {
        let mut t = ExpertTelemetry::new();
        t.record(0, &[0]);
        t.record(0, &[0]); // same expert again
        assert!((t.utilization(0, 8) - 0.125).abs() < f64::EPSILON);
    }

    #[test]
    fn utilization_missing_layer_returns_zero() {
        let t = ExpertTelemetry::new();
        assert!((t.utilization(0, 128) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn utilization_zero_total_experts_is_safe() {
        let t = ExpertTelemetry::new();
        // Should not panic; "total_experts" of 0 is coerced to 1.
        assert!((t.utilization(0, 0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn empty_slice_record_is_noop() {
        let mut t = ExpertTelemetry::new();
        t.record(0, &[]);
        assert_eq!(t.total_steps, 1);
        assert!(t.selections.get(&0).unwrap().is_empty());
    }
}
