//! Generic observability for resident model planning and execution.

use ferrule_common::io_protocol::LoadStage;
use ferrule_model::{ModelInfo, ResidentModelRunner};

use crate::cache::KvPageManagerStats;
use crate::io::{OutputTokenId, RegistryStats, RuntimeMaterializationResolverStats};
use crate::scheduling::{ResourceKind, SequenceSlotPool};
use crate::speculation::SpeculativeMetrics;

use super::{ResidentKvPagePlan, ResidentTopKDriver};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResidentTopKDriverStats {
    pub actions: usize,
    pub prefill_chunks: usize,
    pub prefill_tokens: usize,
    pub decode_steps: usize,
    pub emitted_tokens: usize,
    pub staged_tokens: usize,
    pub finished_sequences: usize,
    pub hard_resource_high_water: Vec<(ResourceKind, u64)>,
    pub speculative: SpeculativeMetrics,
}

impl ResidentTopKDriverStats {
    /// Return monotonic counters since `before`, retaining current gauges.
    pub fn delta_since(&self, before: &Self) -> Self {
        Self {
            actions: self.actions.saturating_sub(before.actions),
            prefill_chunks: self.prefill_chunks.saturating_sub(before.prefill_chunks),
            prefill_tokens: self.prefill_tokens.saturating_sub(before.prefill_tokens),
            decode_steps: self.decode_steps.saturating_sub(before.decode_steps),
            emitted_tokens: self.emitted_tokens.saturating_sub(before.emitted_tokens),
            staged_tokens: self.staged_tokens.saturating_sub(before.staged_tokens),
            finished_sequences: self
                .finished_sequences
                .saturating_sub(before.finished_sequences),
            hard_resource_high_water: self.hard_resource_high_water.clone(),
            speculative: speculative_metrics_delta(&before.speculative, &self.speculative),
        }
    }
}

/// Admission metrics for the resident radix prefix cache.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidentPrefixCacheStats {
    /// Successfully published fresh admissions that reused a cached prefix.
    pub hits: usize,
    /// Successfully published cache-eligible admissions whose lookup found no prefix.
    pub misses: usize,
}

impl ResidentPrefixCacheStats {
    pub fn delta_since(self, before: Self) -> Self {
        Self {
            hits: self.hits.saturating_sub(before.hits),
            misses: self.misses.saturating_sub(before.misses),
        }
    }
}

/// Runtime-owned physical KV capacity and current utilization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidentKvCacheObservability {
    pub page_size_tokens: usize,
    pub full_capacity_pages: usize,
    pub configured_pages: usize,
    pub page_bytes: Option<u64>,
    pub configured_bytes: Option<u64>,
    pub stats: KvPageManagerStats,
}

/// Current operation counts grouped by the runtime materialization state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentMaterializationStageObservability {
    pub stage: LoadStage,
    pub active_operations: usize,
}

/// Critical-path accounting captured when one output token is externally committed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidentExternalTokenObservability {
    pub external_token_id: u64,
    pub externally_committed_tokens: usize,
    pub captured_at_ns: u64,
    pub read_ns: u64,
    pub upload_ns: u64,
    pub publish_ns: u64,
    pub wait_ns: u64,
    pub covered_wait_ns: u64,
    pub uncovered_wait_ns: u64,
    pub resume_ns: u64,
    pub commit_ns: u64,
}

/// Model-neutral runtime materialization snapshot.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResidentMaterializationObservability {
    pub resolver: RuntimeMaterializationResolverStats,
    pub registry: RegistryStats,
    pub active_operations: usize,
    pub active_prefetches: usize,
    pub waiting_dependencies: usize,
    pub runnable_actions: usize,
    pub pending_completions: usize,
    pub pending_physical_operations: usize,
    pub resident_entries: usize,
    pub resident_bytes: u64,
    pub stages: Vec<ResidentMaterializationStageObservability>,
    pub external_tokens: Vec<ResidentExternalTokenObservability>,
}

impl ResidentMaterializationObservability {
    /// Return monotonic counters and external-token samples since `before`.
    /// Current queue, residency, and stage values remain gauges.
    pub fn delta_since(&self, before: &Self) -> Self {
        Self {
            resolver: RuntimeMaterializationResolverStats {
                resolves: self
                    .resolver
                    .resolves
                    .saturating_sub(before.resolver.resolves),
            },
            registry: RegistryStats {
                operations_created: self
                    .registry
                    .operations_created
                    .saturating_sub(before.registry.operations_created),
                single_flight_joins: self
                    .registry
                    .single_flight_joins
                    .saturating_sub(before.registry.single_flight_joins),
                physical_completions: self
                    .registry
                    .physical_completions
                    .saturating_sub(before.registry.physical_completions),
                rejected_completions: self
                    .registry
                    .rejected_completions
                    .saturating_sub(before.registry.rejected_completions),
                publications: self
                    .registry
                    .publications
                    .saturating_sub(before.registry.publications),
                retirements: self
                    .registry
                    .retirements
                    .saturating_sub(before.registry.retirements),
                cancellations_requested: self
                    .registry
                    .cancellations_requested
                    .saturating_sub(before.registry.cancellations_requested),
            },
            active_operations: self.active_operations,
            active_prefetches: self.active_prefetches,
            waiting_dependencies: self.waiting_dependencies,
            runnable_actions: self.runnable_actions,
            pending_completions: self.pending_completions,
            pending_physical_operations: self.pending_physical_operations,
            resident_entries: self.resident_entries,
            resident_bytes: self.resident_bytes,
            stages: self.stages.clone(),
            external_tokens: self
                .external_tokens
                .iter()
                .copied()
                .filter(|sample| {
                    !before
                        .external_tokens
                        .iter()
                        .any(|baseline| baseline.external_token_id == sample.external_token_id)
                })
                .collect(),
        }
    }
}

/// Stable model-neutral snapshot exposed by every resident session engine.
#[derive(Debug, Clone)]
pub struct ResidentEngineObservability {
    pub model: ModelInfo,
    pub driver: ResidentTopKDriverStats,
    pub prefix_cache: ResidentPrefixCacheStats,
    pub kv_cache: Option<ResidentKvCacheObservability>,
    pub materialization: ResidentMaterializationObservability,
}

impl ResidentEngineObservability {
    /// Return work performed since `before`, while retaining current runtime gauges.
    pub fn delta_since(&self, before: &Self) -> Self {
        Self {
            model: self.model.clone(),
            driver: self.driver.delta_since(&before.driver),
            prefix_cache: self.prefix_cache.delta_since(before.prefix_cache),
            kv_cache: self.kv_cache,
            materialization: self.materialization.delta_since(&before.materialization),
        }
    }
}

pub(crate) fn snapshot_driver<R, C>(
    driver: &ResidentTopKDriver<R, C>,
    kv_page_plan: Option<ResidentKvPagePlan>,
) -> ResidentEngineObservability
where
    R: ResidentModelRunner,
    C: SequenceSlotPool,
{
    let registry = driver.load_registry();
    let waiting_dependencies = registry
        .waiters()
        .active_waiters()
        .filter_map(|waiter| registry.waiters().loads_for(waiter))
        .fold(0usize, |total, dependencies| {
            total.saturating_add(dependencies.len())
        });
    let mut driver_stats = driver.stats().clone();
    driver_stats.hard_resource_high_water = registry
        .resources()
        .snapshots()
        .map(|snapshot| (snapshot.kind, snapshot.high_water))
        .collect();

    let mut external_tokens = Vec::new();
    let mut external_token_id = 1u64;
    while let Some(snapshot) = registry
        .ledger()
        .output(OutputTokenId::new(external_token_id))
    {
        external_tokens.push(ResidentExternalTokenObservability {
            external_token_id,
            externally_committed_tokens: snapshot.externally_committed_tokens,
            captured_at_ns: snapshot.captured_at_ns,
            read_ns: snapshot.read_ns,
            upload_ns: snapshot.upload_ns,
            publish_ns: snapshot.publish_ns,
            wait_ns: snapshot.wait_ns,
            covered_wait_ns: snapshot.covered_wait_ns,
            uncovered_wait_ns: snapshot.uncovered_wait_ns,
            resume_ns: snapshot.resume_ns,
            commit_ns: snapshot.commit_ns,
        });
        let Some(next) = external_token_id.checked_add(1) else {
            break;
        };
        external_token_id = next;
    }

    ResidentEngineObservability {
        model: driver.model_info(),
        driver: driver_stats,
        prefix_cache: *driver.prefix_cache_stats(),
        kv_cache: driver.page_manager().map(|manager| {
            let plan = kv_page_plan.unwrap_or(ResidentKvPagePlan {
                full_capacity_pages: manager.max_pages(),
                configured_pages: manager.max_pages(),
                page_bytes: None,
                configured_bytes: None,
            });
            ResidentKvCacheObservability {
                page_size_tokens: manager.page_size(),
                full_capacity_pages: plan.full_capacity_pages,
                configured_pages: plan.configured_pages,
                page_bytes: plan.page_bytes,
                configured_bytes: plan.configured_bytes,
                stats: manager.stats(),
            }
        }),
        materialization: ResidentMaterializationObservability {
            resolver: driver.materialization_resolver_stats(),
            registry: registry.stats(),
            active_operations: registry.active_operations(),
            active_prefetches: registry.active_prefetches(),
            waiting_dependencies,
            runnable_actions: registry.runnable_actions(),
            pending_completions: registry.pending_completions(),
            pending_physical_operations: registry.pending_physical_operations(),
            resident_entries: registry.resident_entries(),
            resident_bytes: registry.resident_bytes(),
            stages: registry
                .stage_counts()
                .map(
                    |(stage, active_operations)| ResidentMaterializationStageObservability {
                        stage,
                        active_operations,
                    },
                )
                .collect(),
            external_tokens,
        },
    }
}

#[derive(Debug, Default)]
pub(crate) struct ResidentDriverObservability {
    pub(crate) stats: ResidentTopKDriverStats,
    pub(crate) prefix_cache: ResidentPrefixCacheStats,
}

impl ResidentDriverObservability {
    pub(crate) fn stats(&self) -> &ResidentTopKDriverStats {
        &self.stats
    }

    pub(crate) fn prefix_cache_stats(&self) -> &ResidentPrefixCacheStats {
        &self.prefix_cache
    }

    pub(crate) fn prefix_hits(&self) -> usize {
        self.prefix_cache.hits
    }

    pub(crate) fn prefix_misses(&self) -> usize {
        self.prefix_cache.misses
    }
}

fn speculative_metrics_delta(
    before: &SpeculativeMetrics,
    after: &SpeculativeMetrics,
) -> SpeculativeMetrics {
    let histogram_len = after
        .accepted_prefix_histogram
        .len()
        .max(before.accepted_prefix_histogram.len());
    SpeculativeMetrics {
        cycles: after.cycles.saturating_sub(before.cycles),
        proposed_tokens: after.proposed_tokens.saturating_sub(before.proposed_tokens),
        verified_rows: after.verified_rows.saturating_sub(before.verified_rows),
        accepted_draft_tokens: after
            .accepted_draft_tokens
            .saturating_sub(before.accepted_draft_tokens),
        correction_tokens: after
            .correction_tokens
            .saturating_sub(before.correction_tokens),
        externally_committed_tokens: after
            .externally_committed_tokens
            .saturating_sub(before.externally_committed_tokens),
        runtime_emitted_tokens: after
            .runtime_emitted_tokens
            .saturating_sub(before.runtime_emitted_tokens),
        rolled_back_rows: after
            .rolled_back_rows
            .saturating_sub(before.rolled_back_rows),
        rejected_tokens: after.rejected_tokens.saturating_sub(before.rejected_tokens),
        accepted_prefix_histogram: (0..histogram_len)
            .map(|index| {
                after
                    .accepted_prefix_histogram
                    .get(index)
                    .copied()
                    .unwrap_or_default()
                    .saturating_sub(
                        before
                            .accepted_prefix_histogram
                            .get(index)
                            .copied()
                            .unwrap_or_default(),
                    )
            })
            .collect(),
        total_proposal_time_us: after
            .total_proposal_time_us
            .saturating_sub(before.total_proposal_time_us),
        total_transaction_time_us: after
            .total_transaction_time_us
            .saturating_sub(before.total_transaction_time_us),
        total_verify_time_us: after
            .total_verify_time_us
            .saturating_sub(before.total_verify_time_us),
        total_cycle_time_us: after
            .total_cycle_time_us
            .saturating_sub(before.total_cycle_time_us),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_deltas_keep_gauges_and_subtract_counters() {
        let before = ResidentMaterializationObservability {
            resolver: RuntimeMaterializationResolverStats { resolves: 7 },
            registry: RegistryStats {
                operations_created: 9,
                retirements: 4,
                ..Default::default()
            },
            active_operations: 8,
            resident_entries: 6,
            external_tokens: vec![ResidentExternalTokenObservability {
                external_token_id: 1,
                ..Default::default()
            }],
            ..Default::default()
        };
        let after = ResidentMaterializationObservability {
            resolver: RuntimeMaterializationResolverStats { resolves: 11 },
            registry: RegistryStats {
                operations_created: 12,
                retirements: 9,
                ..Default::default()
            },
            active_operations: 2,
            resident_entries: 3,
            external_tokens: vec![
                before.external_tokens[0],
                ResidentExternalTokenObservability {
                    external_token_id: 2,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let delta = after.delta_since(&before);
        assert_eq!(delta.resolver.resolves, 4);
        assert_eq!(delta.registry.operations_created, 3);
        assert_eq!(delta.registry.retirements, 5);
        assert_eq!(delta.active_operations, 2);
        assert_eq!(delta.resident_entries, 3);
        assert_eq!(delta.external_tokens.len(), 1);
        assert_eq!(delta.external_tokens[0].external_token_id, 2);
    }

    #[test]
    fn driver_delta_saturates_and_retains_resource_gauges() {
        let before = ResidentTopKDriverStats {
            actions: 10,
            emitted_tokens: 8,
            hard_resource_high_water: vec![(ResourceKind::KvPage, 99)],
            ..Default::default()
        };
        let after = ResidentTopKDriverStats {
            actions: 4,
            emitted_tokens: 12,
            hard_resource_high_water: vec![(ResourceKind::KvPage, 3)],
            ..Default::default()
        };

        let delta = after.delta_since(&before);
        assert_eq!(delta.actions, 0);
        assert_eq!(delta.emitted_tokens, 4);
        assert_eq!(
            delta.hard_resource_high_water,
            after.hard_resource_high_water
        );
    }
}
