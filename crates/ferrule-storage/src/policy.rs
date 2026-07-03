//! Policy — decides what should be resident, staged, or evicted.
//!
//! Named `StorageResidencyPolicy` to avoid collision with `ferrule-model`'s
//! model-level `ResidencyPolicy` (`streaming_allowed` / `all_resident_required`).

/// Runtime-level residency policy: per-tier budgets, eviction weights, prefetch
/// configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct StorageResidencyPolicy {
    pub budgets: Budgets,
    pub retain_hot: bool,
    pub prefetch_window: usize,
    pub eviction_weights: EvictionWeights,
}

impl Default for StorageResidencyPolicy {
    fn default() -> Self {
        Self {
            budgets: Budgets::default(),
            retain_hot: true,
            prefetch_window: 4,
            eviction_weights: EvictionWeights::default(),
        }
    }
}

/// Per-tier capacity budgets.
///
/// Both **bytes** (capacity) and **slots** (correctness) are tracked. Per-layer
/// top-k correctness requires at least `num_experts_per_tok` device slots — a
/// pure byte budget can violate this.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Budgets {
    /// Minimum device slots per layer (correctness: >= num_experts_per_tok).
    pub device_slots_per_layer: usize,
    /// Maximum device memory budget in bytes.
    pub device_budget_bytes: u64,
    /// Maximum host staging budget in bytes.
    pub host_staging_budget_bytes: u64,
    /// Maximum local disk cache budget in bytes (None = unlimited).
    pub local_disk_cache_budget_bytes: Option<u64>,
}

impl Default for Budgets {
    fn default() -> Self {
        Self {
            device_slots_per_layer: 8,
            device_budget_bytes: 8 * 1024 * 1024 * 1024, // 8 GiB
            host_staging_budget_bytes: 4 * 1024 * 1024 * 1024, // 4 GiB
            local_disk_cache_budget_bytes: None,
        }
    }
}

/// Weights for the eviction scoring function.
///
/// The policy evicts objects that are cold by **both** recency and frequency
/// (F2 insight). These weights control how recency, frequency, and load cost
/// combine into an eviction score.
#[derive(Debug, Clone, PartialEq)]
pub struct EvictionWeights {
    pub recency: f32,
    pub frequency: f32,
    pub load_cost: f32,
}

impl Default for EvictionWeights {
    fn default() -> Self {
        Self {
            recency: 1.0,
            frequency: 1.0,
            load_cost: 0.5,
        }
    }
}

/// Per-object residency score used by the policy to rank objects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidencyScore {
    /// Must be device-resident right now (selected by router).
    pub execute_now: bool,
    /// Likely needed soon (prefetch candidate).
    pub predicted: bool,
    /// Last step this object was used.
    pub last_used_step: u64,
    /// Cumulative activation count (F2 frequency counter).
    pub activation_count: u64,
    /// How expensive is a miss? (bytes that must be re-read).
    pub load_cost_bytes: u64,
    /// Object size in bytes.
    pub object_bytes: u64,
}

impl ResidencyScore {
    /// Compute a composite eviction priority. Higher = more likely to be
    /// evicted (colder). Objects with `execute_now` are never evicted.
    ///
    /// This is a simple weighted sum. Phase 3 can replace it with a more
    /// sophisticated function once counters are measured.
    pub fn eviction_score(&self, weights: &EvictionWeights, current_step: u64) -> f64 {
        if self.execute_now {
            return f64::MIN; // never evict
        }
        let recency_penalty = (current_step.saturating_sub(self.last_used_step)) as f64;
        let freq_bonus = self.activation_count as f64;
        let cost_factor = self.load_cost_bytes as f64 / self.object_bytes.max(1) as f64;

        // Higher score = colder = better eviction candidate.
        // Recency penalty pushes score up (cold by time).
        // Frequency bonus pushes score down (hot by count).
        // Load cost pushes score down (expensive to reload).
        (weights.recency as f64) * recency_penalty
            - (weights.frequency as f64) * freq_bonus
            - (weights.load_cost as f64) * cost_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_values() {
        let policy = StorageResidencyPolicy::default();
        assert!(policy.retain_hot);
        assert_eq!(policy.prefetch_window, 4);
        assert_eq!(policy.budgets.device_slots_per_layer, 8);
    }

    #[test]
    fn execute_now_never_evicted() {
        let score = ResidencyScore {
            execute_now: true,
            predicted: false,
            last_used_step: 0,
            activation_count: 0,
            load_cost_bytes: 0,
            object_bytes: 4096,
        };
        let weights = EvictionWeights::default();
        assert_eq!(score.eviction_score(&weights, 100), f64::MIN);
    }

    #[test]
    fn cold_object_has_higher_eviction_score() {
        let weights = EvictionWeights::default();
        let cold = ResidencyScore {
            execute_now: false,
            predicted: false,
            last_used_step: 0,
            activation_count: 0,
            load_cost_bytes: 0,
            object_bytes: 4096,
        };
        let hot = ResidencyScore {
            execute_now: false,
            predicted: false,
            last_used_step: 95,
            activation_count: 50,
            load_cost_bytes: 4096,
            object_bytes: 4096,
        };
        let current_step = 100;
        assert!(
            cold.eviction_score(&weights, current_step)
                > hot.eviction_score(&weights, current_step),
            "cold object should have higher eviction score (more evictable)"
        );
    }

    #[test]
    fn high_frequency_lowers_eviction_score() {
        let weights = EvictionWeights::default();
        let rare = ResidencyScore {
            execute_now: false,
            predicted: false,
            last_used_step: 50,
            activation_count: 1,
            load_cost_bytes: 0,
            object_bytes: 4096,
        };
        let frequent = ResidencyScore {
            execute_now: false,
            predicted: false,
            last_used_step: 50,
            activation_count: 100,
            load_cost_bytes: 0,
            object_bytes: 4096,
        };
        let current_step = 100;
        assert!(
            frequent.eviction_score(&weights, current_step)
                < rare.eviction_score(&weights, current_step),
            "frequent object should have lower eviction score (less evictable)"
        );
    }

    #[test]
    fn high_load_cost_lowers_eviction_score() {
        let weights = EvictionWeights::default();
        let cheap = ResidencyScore {
            execute_now: false,
            predicted: false,
            last_used_step: 50,
            activation_count: 10,
            load_cost_bytes: 0,
            object_bytes: 4096,
        };
        let expensive = ResidencyScore {
            execute_now: false,
            predicted: false,
            last_used_step: 50,
            activation_count: 10,
            load_cost_bytes: 4096,
            object_bytes: 4096,
        };
        let current_step = 100;
        assert!(
            expensive.eviction_score(&weights, current_step)
                < cheap.eviction_score(&weights, current_step),
            "expensive-to-reload object should have lower eviction score"
        );
    }

    #[test]
    fn budgets_default_reasonable() {
        let b = Budgets::default();
        assert!(b.device_budget_bytes > 0);
        assert!(b.host_staging_budget_bytes > 0);
        assert!(b.local_disk_cache_budget_bytes.is_none());
    }
}
