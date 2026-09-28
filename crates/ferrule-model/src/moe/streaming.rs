//! Expert streaming and residency planning.
//!
//! This module is intentionally model-family agnostic. A model adapter decides
//! which artifact tensors represent an expert; the runtime decides when an expert
//! should be GPU-resident, prefetched, evicted, or streamed from a slower tier.
//! Quality-first adapters can preserve artifact FP4/FP8 payloads and stream experts
//! instead of immediately re-quantizing them to fit.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::sync::{Arc, mpsc};

use crate::HfRoutedExpertTensorInfo;
use ferrule_common::{Error, MemoryPoolLimits, MemoryPoolStats, OwnerMemoryLru, Result};

// Keep the historical streaming API while separating semantic payloads from I/O.
#[path = "source.rs"]
pub(super) mod source;
#[path = "storage.rs"]
pub(super) mod storage;

pub use source::{
    ExpertArtifactPayload, ExpertBundleFormat, ExpertBundleLayout, ExpertComputeBundle, ExpertId,
    ExpertLinearFormat, ExpertLinearPayload, ExpertLoadSource, ExpertMatrixKind,
    ExpertSourceCatalog, ExpertSourceDenseLayout, ExpertStorageTier, ExpertTensorComponent,
    ExpertTensorKey, ExpertTensorPayload, ExpertTensorSlice,
};
#[allow(unused_imports)] // Compatibility exports also exist in non-CUDA builds.
pub(crate) use source::{infer_expert_linear_format, typed_expert_bundle_layout};
#[cfg(all(target_os = "linux", feature = "cuda"))]
#[allow(unused_imports)] // Preserve the existing crate-private return type path.
pub(crate) use storage::ReservedPinnedExpertLoad;
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub(crate) use storage::{
    ExpertIoPlan, PinnedExpertArtifactPayload, PinnedExpertLoadPlan, PinnedExpertReadPoll,
    PinnedExpertReadTicket,
};
pub use storage::{
    ExpertIoStats, ExpertIoTransport, ExpertIoTransportError, ExpertStreamingReader,
    read_experts_concurrent,
};

/// Per-layer routed-expert source catalog and its streaming policy.
#[derive(Debug, Clone)]
pub struct ExpertLayerSources {
    source_catalog: std::sync::Arc<ExpertSourceCatalog>,
    streaming_policy: ExpertStreamingPolicy,
}

impl ExpertLayerSources {
    pub fn new(
        source_catalog: std::sync::Arc<ExpertSourceCatalog>,
        streaming_policy: ExpertStreamingPolicy,
    ) -> Self {
        Self {
            source_catalog,
            streaming_policy,
        }
    }

    pub fn source_catalog(&self) -> &std::sync::Arc<ExpertSourceCatalog> {
        &self.source_catalog
    }

    pub const fn streaming_policy(&self) -> &ExpertStreamingPolicy {
        &self.streaming_policy
    }

    pub const fn resident_capacity(&self) -> usize {
        self.streaming_policy.gpu_slots_per_layer
    }

    pub const fn prefetch_capacity(&self) -> usize {
        self.streaming_policy.prefetch_per_layer
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertLoadReason {
    Selected,
    Prefetch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertLoadRequest {
    pub expert: ExpertId,
    pub load_source: ExpertLoadSource,
    pub reason: ExpertLoadReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertEvictRequest {
    pub expert: ExpertId,
    pub target: ExpertStorageTier,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertStreamingPolicy {
    /// Maximum concurrently GPU-resident experts per layer.
    ///
    /// For top-k MoE decode this must be at least `num_experts_per_tok` unless a
    /// later executor implements sequential per-expert load/compute/evict.
    pub gpu_slots_per_layer: usize,
    /// Best-effort predicted experts to load after selected experts are covered.
    pub prefetch_per_layer: usize,
    /// Keep the exact source payload encoding; do not force a conversion policy here.
    pub preserve_source_encoding: bool,
    /// Whether CPU RAM staging is allowed. Disabling this models very constrained
    /// hosts where streaming should go directly from local/remote chunks.
    pub allow_cpu_staging: bool,
    /// Whether remote/object/LAN sources may satisfy expert loads.
    pub allow_remote_sources: bool,
}

impl ExpertStreamingPolicy {
    pub fn quality_first(num_experts_per_tok: usize) -> Self {
        let gpu_slots_per_layer = num_experts_per_tok
            .saturating_mul(2)
            .max(num_experts_per_tok);
        Self {
            gpu_slots_per_layer,
            prefetch_per_layer: num_experts_per_tok,
            preserve_source_encoding: true,
            allow_cpu_staging: false,
            allow_remote_sources: false,
        }
    }

    pub fn quality_first_no_prefetch(num_experts_per_tok: usize) -> Self {
        Self {
            gpu_slots_per_layer: num_experts_per_tok
                .saturating_mul(8)
                .max(num_experts_per_tok),
            prefetch_per_layer: 0,
            ..Self::quality_first(num_experts_per_tok)
        }
    }

    pub fn quality_first_with_prefetch(
        num_experts_per_tok: usize,
        prefetch_per_layer: usize,
    ) -> Self {
        let no_prefetch_slots = num_experts_per_tok
            .saturating_mul(8)
            .max(num_experts_per_tok);
        let gpu_slots_per_layer = no_prefetch_slots.max(
            num_experts_per_tok
                .saturating_add(prefetch_per_layer)
                .max(num_experts_per_tok),
        );
        Self {
            gpu_slots_per_layer,
            prefetch_per_layer,
            ..Self::quality_first(num_experts_per_tok)
        }
    }

    pub fn quality_first_remote(num_experts_per_tok: usize) -> Self {
        Self {
            allow_remote_sources: true,
            ..Self::quality_first(num_experts_per_tok)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertStreamingStep {
    pub layer: usize,
    pub selected: Vec<ExpertId>,
    pub prefetched: Vec<ExpertId>,
    pub loads: Vec<ExpertLoadRequest>,
    pub evictions: Vec<ExpertEvictRequest>,
}

impl ExpertStreamingStep {
    pub fn is_noop(&self) -> bool {
        self.loads.is_empty() && self.evictions.is_empty()
    }
}

#[derive(Debug, Clone)]
struct ExpertState {
    location: ExpertStorageTier,
    last_used_step: u64,
    selected_count: u64,
}

#[derive(Debug, Clone)]
pub struct ExpertStreamingPlanner {
    policy: ExpertStreamingPolicy,
    source_catalog: Arc<ExpertSourceCatalog>,
    experts: BTreeMap<ExpertId, ExpertState>,
    step: u64,
}

impl ExpertStreamingPlanner {
    pub fn new(policy: ExpertStreamingPolicy) -> Self {
        Self::from_catalog(policy, Arc::new(ExpertSourceCatalog::default()))
    }

    pub fn from_catalog(
        policy: ExpertStreamingPolicy,
        source_catalog: Arc<ExpertSourceCatalog>,
    ) -> Self {
        let experts = source_catalog
            .iter()
            .map(|(expert, source)| {
                (
                    *expert,
                    ExpertState {
                        location: source.tier(),
                        last_used_step: 0,
                        selected_count: 0,
                    },
                )
            })
            .collect();
        Self {
            policy,
            source_catalog,
            experts,
            step: 0,
        }
    }

    pub fn policy(&self) -> &ExpertStreamingPolicy {
        &self.policy
    }

    pub fn source_catalog(&self) -> &Arc<ExpertSourceCatalog> {
        &self.source_catalog
    }

    /// Compatibility registration for incremental callers.
    ///
    /// The currently shared catalog is never mutated. Instead, registration
    /// replaces this planner's catalog with a new immutable snapshot.
    pub fn register_load_source(&mut self, expert: ExpertId, load_source: ExpertLoadSource) {
        let location = load_source.tier();
        let mut sources = self
            .source_catalog
            .iter_entries()
            .map(|(expert, source, resource_source)| (*expert, (source.clone(), resource_source)))
            .collect::<BTreeMap<_, _>>();
        sources.insert(expert, (load_source, None));
        self.source_catalog =
            Arc::new(ExpertSourceCatalog::from_entries(sources.into_iter().map(
                |(expert, (source, resource_source))| (expert, source, resource_source),
            )));
        self.experts.insert(
            expert,
            ExpertState {
                location,
                last_used_step: 0,
                selected_count: 0,
            },
        );
    }

    pub fn register_hf_routed_expert_tensor_sets(
        &mut self,
        model_dir: &Path,
        tensors: impl IntoIterator<Item = HfRoutedExpertTensorInfo>,
    ) -> Result<usize> {
        let catalog = ExpertSourceCatalog::from_hf_routed_expert_tensor_sets(model_dir, tensors)?;
        let count = catalog.count();
        let mut sources = self
            .source_catalog
            .iter_entries()
            .map(|(expert, source, resource_source)| (*expert, (source.clone(), resource_source)))
            .collect::<BTreeMap<_, _>>();
        for (expert, source, resource_source) in catalog.iter_entries() {
            sources.insert(*expert, (source.clone(), resource_source));
            self.experts.insert(
                *expert,
                ExpertState {
                    location: source.tier(),
                    last_used_step: 0,
                    selected_count: 0,
                },
            );
        }
        self.source_catalog =
            Arc::new(ExpertSourceCatalog::from_entries(sources.into_iter().map(
                |(expert, (source, resource_source))| (expert, source, resource_source),
            )));
        Ok(count)
    }

    pub fn mark_resident(&mut self, expert: ExpertId, location: ExpertStorageTier) -> Result<()> {
        let state = self.experts.get_mut(&expert).ok_or_else(|| Error::Model {
            message: format!(
                "expert streaming load source missing for layer {} expert {}",
                expert.layer, expert.expert
            ),
        })?;
        state.location = location;
        Ok(())
    }

    pub fn location(&self, expert: ExpertId) -> Option<ExpertStorageTier> {
        self.experts.get(&expert).map(|state| state.location)
    }

    pub fn resident_experts(&self, layer: usize) -> Vec<ExpertId> {
        self.experts
            .iter()
            .filter_map(|(expert, state)| {
                (expert.layer == layer && state.location.is_gpu_ready()).then_some(*expert)
            })
            .collect()
    }

    fn resident_experts_by_hotness(&self, layer: usize) -> Vec<ExpertId> {
        let mut experts = self
            .experts
            .iter()
            .filter_map(|(expert, state)| {
                (expert.layer == layer && state.location.is_gpu_ready()).then_some((
                    *expert,
                    state.selected_count,
                    state.last_used_step,
                ))
            })
            .collect::<Vec<_>>();
        experts.sort_by(
            |(left, left_count, left_step), (right, right_count, right_step)| {
                right_count
                    .cmp(left_count)
                    .then_with(|| right_step.cmp(left_step))
                    .then_with(|| left.expert.cmp(&right.expert))
            },
        );
        experts.into_iter().map(|(expert, _, _)| expert).collect()
    }

    /// Per-layer routing-aware hotset, ordered from hottest to coldest.
    ///
    /// This intentionally only returns experts that have actually been selected
    /// before. It avoids the old naive `0..N` prefetch pattern, which is not
    /// correlated with DSV4 routing and can increase residency churn.
    pub fn hot_experts(&self, layer: usize, count: usize) -> Vec<usize> {
        if count == 0 {
            return Vec::new();
        }
        let mut experts = self
            .experts
            .iter()
            .filter_map(|(expert, state)| {
                (expert.layer == layer && state.selected_count > 0).then_some((
                    expert.expert,
                    state.selected_count,
                    state.last_used_step,
                ))
            })
            .collect::<Vec<_>>();
        experts.sort_by(
            |(left, left_count, left_step), (right, right_count, right_step)| {
                right_count
                    .cmp(left_count)
                    .then_with(|| right_step.cmp(left_step))
                    .then_with(|| left.cmp(right))
            },
        );
        experts
            .into_iter()
            .take(count)
            .map(|(expert, _, _)| expert)
            .collect()
    }

    pub fn plan_layer_step(
        &mut self,
        layer: usize,
        selected: &[usize],
        predicted: &[usize],
    ) -> Result<ExpertStreamingStep> {
        self.step = self.step.saturating_add(1);
        let selected = unique_ids(layer, selected);
        if selected.len() > self.policy.gpu_slots_per_layer {
            return Err(Error::Model {
                message: format!(
                    "expert streaming policy has {} GPU slots for layer {}, but {} selected experts must be available",
                    self.policy.gpu_slots_per_layer,
                    layer,
                    selected.len()
                ),
            });
        }
        for expert in &selected {
            if let Some(state) = self.experts.get_mut(expert) {
                state.selected_count = state.selected_count.saturating_add(1);
                state.last_used_step = self.step;
            }
        }

        let mut target = selected.iter().copied().collect::<BTreeSet<_>>();
        let mut prefetched = Vec::new();
        for expert in unique_ids(layer, predicted) {
            if target.contains(&expert) {
                continue;
            }
            if prefetched.len() >= self.policy.prefetch_per_layer {
                break;
            }
            if target.len() >= self.policy.gpu_slots_per_layer {
                break;
            }
            target.insert(expert);
            prefetched.push(expert);
        }

        let mut current_gpu = self.resident_experts_by_hotness(layer);
        for expert in current_gpu.iter().copied() {
            if target.len() >= self.policy.gpu_slots_per_layer {
                break;
            }
            target.insert(expert);
        }

        let mut evictions = Vec::new();
        for expert in current_gpu.drain(..) {
            if !target.contains(&expert) {
                evictions.push(ExpertEvictRequest {
                    expert,
                    target: self.load_source_tier_or_local(expert)?,
                });
            }
        }

        let mut loads = Vec::new();
        for expert in &selected {
            self.ensure_load_source_allowed(*expert)?;
            if !matches!(self.location(*expert), Some(ExpertStorageTier::Gpu)) {
                loads.push(ExpertLoadRequest {
                    expert: *expert,
                    load_source: self.load_source_for(*expert)?.clone(),
                    reason: ExpertLoadReason::Selected,
                });
            }
        }
        for expert in &prefetched {
            self.ensure_load_source_allowed(*expert)?;
            if !matches!(self.location(*expert), Some(ExpertStorageTier::Gpu)) {
                loads.push(ExpertLoadRequest {
                    expert: *expert,
                    load_source: self.load_source_for(*expert)?.clone(),
                    reason: ExpertLoadReason::Prefetch,
                });
            }
        }

        Ok(ExpertStreamingStep {
            layer,
            selected,
            prefetched,
            loads,
            evictions,
        })
    }

    pub fn commit_step(&mut self, step: &ExpertStreamingStep) -> Result<()> {
        self.commit_step_loaded(step, step.loads.iter().map(|load| load.expert))
    }

    /// Commit a planner step after the backend has only materialized a subset of
    /// requested loads.
    ///
    /// This is useful for latency-oriented backends that enqueue `Prefetch` loads
    /// asynchronously: selected experts are still committed as GPU-resident for
    /// correctness, while queued-but-not-ready prefetches remain at their source
    /// tier until a later step actually consumes/uploads them.
    pub fn commit_step_loaded(
        &mut self,
        step: &ExpertStreamingStep,
        loaded: impl IntoIterator<Item = ExpertId>,
    ) -> Result<()> {
        for eviction in &step.evictions {
            if self.experts.contains_key(&eviction.expert) {
                self.mark_resident(eviction.expert, eviction.target)?;
            }
        }
        for expert in loaded {
            if self.experts.contains_key(&expert) {
                self.mark_resident(expert, ExpertStorageTier::Gpu)?;
            }
        }
        Ok(())
    }

    fn load_source_for(&self, expert: ExpertId) -> Result<&ExpertLoadSource> {
        self.source_catalog
            .source(expert)
            .ok_or_else(|| Error::Model {
                message: format!(
                    "expert streaming load source missing for layer {} expert {}",
                    expert.layer, expert.expert
                ),
            })
    }

    fn load_source_tier_or_local(&self, expert: ExpertId) -> Result<ExpertStorageTier> {
        let tier = self.load_source_for(expert)?.tier();
        Ok(if tier == ExpertStorageTier::Gpu {
            ExpertStorageTier::LocalStorage
        } else {
            tier
        })
    }

    fn ensure_load_source_allowed(&self, expert: ExpertId) -> Result<()> {
        let load_source = self.load_source_for(expert)?;
        let tier = load_source.tier();
        if !tier.is_streamable() && tier != ExpertStorageTier::Gpu {
            return Err(Error::Model {
                message: format!(
                    "expert load source for layer {} expert {} is not streamable: {:?}",
                    expert.layer, expert.expert, tier
                ),
            });
        }
        if tier == ExpertStorageTier::Remote && !self.policy.allow_remote_sources {
            return Err(Error::Model {
                message: format!(
                    "remote expert load source for layer {} expert {} requires allow_remote_sources=true",
                    expert.layer, expert.expert
                ),
            });
        }
        if tier == ExpertStorageTier::Cpu && !self.policy.allow_cpu_staging {
            return Err(Error::Model {
                message: format!(
                    "CPU-staged expert load source for layer {} expert {} requires allow_cpu_staging=true",
                    expert.layer, expert.expert
                ),
            });
        }
        Ok(())
    }
}

fn unique_ids(layer: usize, experts: &[usize]) -> Vec<ExpertId> {
    experts
        .iter()
        .copied()
        .map(|expert| ExpertId::new(layer, expert))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

// ── HostStagedExpertCache ──────────────────────────────────────────────────

/// Model-family-neutral retention policy for streamed expert payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertMemoryPolicy {
    pub host_staged: MemoryPoolLimits,
    pub pinned_host: MemoryPoolLimits,
}

impl ExpertMemoryPolicy {
    pub const fn new(host_staged: MemoryPoolLimits, pinned_host: MemoryPoolLimits) -> Self {
        Self {
            host_staged,
            pinned_host,
        }
    }
}

impl Default for ExpertMemoryPolicy {
    fn default() -> Self {
        Self::new(
            MemoryPoolLimits::entries_only(256),
            MemoryPoolLimits::entries_only(64),
        )
    }
}

/// Owner-thread LRU cache of complete expert compute bundles staged in host RAM.
///
/// Bundles are shared through [`Arc`], so cache hits do not copy tensor payloads.
/// The cache has no internal synchronization: one owner thread performs all
/// mutations. Resident bytes and both ends of the LRU list are maintained
/// incrementally, making accounting, hits, and eviction O(1) on average.
#[derive(Debug)]
pub struct HostStagedExpertCache {
    cache: OwnerMemoryLru<ExpertId, Arc<ExpertComputeBundle>>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AsyncHostStagedExpertStats {
    pub submitted: u64,
    pub completed: u64,
    pub failed: u64,
    pub skipped: u64,
    pub in_flight: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertResidencySelectedLoad {
    pub expert: ExpertId,
    pub load_source: ExpertLoadSource,
    pub host_staged: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertResidencyPrefetchLoad {
    pub expert: ExpertId,
    pub load_source: ExpertLoadSource,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExpertResidencyPlan {
    pub selected_resident: Vec<ExpertId>,
    pub selected_materializing: Vec<ExpertId>,
    pub selected_host_staged: Vec<ExpertResidencySelectedLoad>,
    /// Selected expert whose async host-staging read is already in flight.
    /// Backends should wait for and reuse that read before falling back to a
    /// duplicate synchronous disk read.
    pub selected_in_flight: Vec<ExpertResidencySelectedLoad>,
    pub selected_cold: Vec<ExpertResidencySelectedLoad>,
    pub prefetch_resident: Vec<ExpertId>,
    pub prefetch_materializing: Vec<ExpertId>,
    pub prefetch_host_staged: Vec<ExpertId>,
    pub prefetch_in_flight: Vec<ExpertId>,
    pub prefetch_cold: Vec<ExpertResidencyPrefetchLoad>,
}

impl ExpertResidencyPlan {
    pub fn selected_to_materialize(&self) -> impl Iterator<Item = &ExpertResidencySelectedLoad> {
        self.selected_host_staged.iter().chain(&self.selected_cold)
    }

    pub fn selected_waiting_for_host_staging(
        &self,
    ) -> impl Iterator<Item = &ExpertResidencySelectedLoad> {
        self.selected_in_flight.iter()
    }

    pub fn selected_miss_count(&self) -> usize {
        self.selected_materializing.len()
            + self.selected_host_staged.len()
            + self.selected_in_flight.len()
            + self.selected_cold.len()
    }

    pub fn selected_resident_count(&self) -> usize {
        self.selected_resident.len()
    }

    pub fn prefetch_load_count(&self) -> usize {
        self.prefetch_resident.len()
            + self.prefetch_materializing.len()
            + self.prefetch_host_staged.len()
            + self.prefetch_in_flight.len()
            + self.prefetch_cold.len()
    }

    pub fn prefetch_enqueue_count(&self) -> usize {
        self.prefetch_cold.len()
    }

    pub fn prefetch_skipped_cached_or_inflight_count(&self) -> usize {
        self.prefetch_resident.len()
            + self.prefetch_materializing.len()
            + self.prefetch_in_flight.len()
    }
}

pub fn classify_expert_residency(
    loads: &[ExpertLoadRequest],
    is_gpu_resident: impl Fn(ExpertId) -> bool,
    is_materializing: impl Fn(ExpertId) -> bool,
    is_host_staged: impl Fn(ExpertId) -> bool,
    is_in_flight: impl Fn(ExpertId) -> bool,
) -> ExpertResidencyPlan {
    let mut plan = ExpertResidencyPlan::default();
    for load in loads {
        let expert = load.expert;
        match load.reason {
            ExpertLoadReason::Selected => {
                if is_gpu_resident(expert) {
                    plan.selected_resident.push(expert);
                } else if is_materializing(expert) {
                    plan.selected_materializing.push(expert);
                } else if is_host_staged(expert) {
                    plan.selected_host_staged.push(ExpertResidencySelectedLoad {
                        expert,
                        load_source: load.load_source.clone(),
                        host_staged: true,
                    });
                } else if is_in_flight(expert) {
                    plan.selected_in_flight.push(ExpertResidencySelectedLoad {
                        expert,
                        load_source: load.load_source.clone(),
                        host_staged: false,
                    });
                } else {
                    plan.selected_cold.push(ExpertResidencySelectedLoad {
                        expert,
                        load_source: load.load_source.clone(),
                        host_staged: false,
                    });
                }
            }
            ExpertLoadReason::Prefetch => {
                if is_gpu_resident(expert) {
                    plan.prefetch_resident.push(expert);
                } else if is_materializing(expert) {
                    plan.prefetch_materializing.push(expert);
                } else if is_host_staged(expert) {
                    plan.prefetch_host_staged.push(expert);
                } else if is_in_flight(expert) {
                    plan.prefetch_in_flight.push(expert);
                } else {
                    plan.prefetch_cold.push(ExpertResidencyPrefetchLoad {
                        expert,
                        load_source: load.load_source.clone(),
                    });
                }
            }
        }
    }
    plan
}

enum AsyncHostStagedExpertResult {
    Loaded(Box<ExpertComputeBundle>),
    Failed { expert: ExpertId, error: Error },
}

impl AsyncHostStagedExpertResult {
    fn expert(&self) -> ExpertId {
        match self {
            Self::Loaded(bundle) => bundle.expert,
            Self::Failed { expert, .. } => *expert,
        }
    }
}

/// Best-effort asynchronous host staging for expert payloads.
///
/// The loader intentionally stops at host RAM. CUDA contexts/streams remain owned
/// by the main thread; background workers only fault/read safetensors slices and
/// build `ExpertComputeBundle`s. Main-thread MoE code drains completed bundles
/// into `HostStagedExpertCache` before it decides whether a selected expert must
/// synchronously read from disk.
pub struct AsyncHostStagedExpertLoader {
    tx: mpsc::Sender<AsyncHostStagedExpertResult>,
    rx: mpsc::Receiver<AsyncHostStagedExpertResult>,
    in_flight: BTreeSet<ExpertId>,
    max_in_flight: usize,
    submitted: u64,
    completed: u64,
    failed: u64,
    skipped: u64,
}

impl AsyncHostStagedExpertLoader {
    pub fn new(max_in_flight: usize) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            tx,
            rx,
            in_flight: BTreeSet::new(),
            max_in_flight,
            submitted: 0,
            completed: 0,
            failed: 0,
            skipped: 0,
        }
    }

    pub fn stats(&self) -> AsyncHostStagedExpertStats {
        AsyncHostStagedExpertStats {
            submitted: self.submitted,
            completed: self.completed,
            failed: self.failed,
            skipped: self.skipped,
            in_flight: self.in_flight.len(),
        }
    }

    pub fn is_in_flight(&self, expert: ExpertId) -> bool {
        self.in_flight.contains(&expert)
    }

    pub fn enqueue(
        &mut self,
        expert: ExpertId,
        source: ExpertLoadSource,
        reader: &ExpertStreamingReader,
    ) -> bool {
        if self.max_in_flight == 0
            || self.in_flight.len() >= self.max_in_flight
            || self.in_flight.contains(&expert)
        {
            self.skipped = self.skipped.saturating_add(1);
            return false;
        }
        self.in_flight.insert(expert);
        self.submitted = self.submitted.saturating_add(1);
        let tx = self.tx.clone();
        let completion_hub = reader.completion_hub();
        let reader = reader.clone();
        rayon::spawn(move || {
            let result = reader
                .read_load_source_concurrent(expert, &source)
                .and_then(ExpertComputeBundle::from_artifact_payload);
            let message = match result {
                Ok(bundle) => AsyncHostStagedExpertResult::Loaded(Box::new(bundle)),
                Err(error) => AsyncHostStagedExpertResult::Failed { expert, error },
            };
            let _ = tx.send(message);
            completion_hub.notify();
        });
        true
    }

    pub fn drain_into(
        &mut self,
        cache: &mut HostStagedExpertCache,
        unretained: &mut HashMap<ExpertId, Arc<ExpertComputeBundle>>,
    ) -> usize {
        let mut completed_now = 0usize;
        while let Ok(result) = self.rx.try_recv() {
            if self.handle_result(result, cache, unretained) {
                completed_now += 1;
            }
        }
        completed_now
    }

    /// Wait for a specific in-flight host-staging read and move all completed
    /// bundles observed while waiting into the host cache or the caller-owned
    /// one-shot handoff. A successful read is returned even when long-term cache
    /// admission rejects it.
    pub fn wait_for_into(
        &mut self,
        expert: ExpertId,
        cache: &mut HostStagedExpertCache,
        unretained: &mut HashMap<ExpertId, Arc<ExpertComputeBundle>>,
    ) -> Result<Option<Arc<ExpertComputeBundle>>> {
        if let Some(bundle) = unretained.remove(&expert) {
            return Ok(Some(bundle));
        }
        if !self.in_flight.contains(&expert) {
            return Ok(None);
        }
        while self.in_flight.contains(&expert) {
            match self.rx.recv() {
                Ok(result) => {
                    let completed_expert = result.expert();
                    let loaded = self.handle_result(result, cache, unretained);
                    if completed_expert == expert {
                        if !loaded {
                            return Ok(None);
                        }
                        return Ok(unretained.remove(&expert).or_else(|| cache.get(expert)));
                    }
                }
                Err(_) => {
                    self.in_flight.remove(&expert);
                    self.failed = self.failed.saturating_add(1);
                    return Ok(None);
                }
            }
        }
        Ok(unretained.remove(&expert).or_else(|| cache.get(expert)))
    }

    fn handle_result(
        &mut self,
        result: AsyncHostStagedExpertResult,
        cache: &mut HostStagedExpertCache,
        unretained: &mut HashMap<ExpertId, Arc<ExpertComputeBundle>>,
    ) -> bool {
        match result {
            AsyncHostStagedExpertResult::Loaded(bundle) => {
                self.in_flight.remove(&bundle.expert);
                let bundle = Arc::from(bundle);
                if !cache.insert_shared(Arc::clone(&bundle)) {
                    unretained.insert(bundle.expert, bundle);
                }
                self.completed = self.completed.saturating_add(1);
                true
            }
            AsyncHostStagedExpertResult::Failed { expert, error } => {
                self.in_flight.remove(&expert);
                self.failed = self.failed.saturating_add(1);
                tracing::debug!(
                    layer = expert.layer,
                    expert = expert.expert,
                    %error,
                    "async expert host staging failed"
                );
                false
            }
        }
    }
}

impl Default for AsyncHostStagedExpertLoader {
    fn default() -> Self {
        Self::new(64)
    }
}

impl HostStagedExpertCache {
    /// Create an entry-limited cache, preserving the original constructor API.
    /// Use [`Self::with_limits`] to enforce a host-memory byte budget as well.
    pub fn new(max_entries: usize) -> Self {
        Self::with_limits(MemoryPoolLimits::entries_only(max_entries))
    }

    /// Create a cache that enforces entry and byte limits simultaneously.
    pub fn with_limits(limits: MemoryPoolLimits) -> Self {
        Self {
            cache: OwnerMemoryLru::new(limits),
        }
    }

    /// Look up a staged bundle and mark it most recently used.
    ///
    /// The returned [`Arc`] shares the exact allocation held by the cache; no
    /// expert tensor payload is copied.
    pub fn get(&mut self, expert: ExpertId) -> Option<Arc<ExpertComputeBundle>> {
        self.cache.get_cloned(expert)
    }

    pub fn contains(&self, expert: ExpertId) -> bool {
        self.cache.contains(expert)
    }

    pub fn expert_ids(&self) -> impl Iterator<Item = ExpertId> + '_ {
        self.cache.keys()
    }

    pub fn expert_ids_for_layer(&self, layer: usize) -> Vec<usize> {
        let mut experts = self
            .expert_ids()
            .filter(|expert| expert.layer == layer)
            .map(|expert| expert.expert)
            .collect::<Vec<_>>();
        experts.sort_unstable();
        experts
    }

    /// Insert an owned bundle. Returns `true` when it was admitted.
    pub fn insert(&mut self, bundle: ExpertComputeBundle) -> bool {
        self.insert_shared(Arc::new(bundle))
    }

    /// Insert an already shared bundle. Returns `true` when it was admitted.
    ///
    /// A bundle larger than `max_bytes` (or any bundle when `max_entries` is
    /// zero) is rejected without evicting or replacing existing entries.
    pub fn insert_shared(&mut self, bundle: Arc<ExpertComputeBundle>) -> bool {
        let expert = bundle.expert;
        let bytes = bundle.total_bytes();
        self.cache.insert(expert, bundle, bytes)
    }

    /// Number of currently staged bundles.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn limits(&self) -> MemoryPoolLimits {
        self.cache.limits()
    }

    pub fn stats(&self) -> MemoryPoolStats {
        self.cache.stats()
    }

    /// Cache hit count (experts served from host memory).
    pub fn hits(&self) -> u64 {
        self.cache.stats().hits
    }

    /// Cache miss count (experts that required a disk read).
    pub fn misses(&self) -> u64 {
        self.cache.stats().misses
    }

    /// Number of entries removed to satisfy cache limits.
    pub fn evictions(&self) -> u64 {
        self.cache.stats().evictions
    }

    /// Number of entries rejected by cache limits.
    pub fn rejections(&self) -> u64 {
        self.cache.stats().rejections
    }

    /// Total payload bytes of all staged bundles, maintained in O(1).
    pub fn total_bytes(&self) -> u64 {
        self.cache.stats().bytes_used
    }
}

impl Default for HostStagedExpertCache {
    fn default() -> Self {
        // Compatibility default: 256 entries and no additional byte cap.
        Self::new(256)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn quality_first_policy_preserves_source_encoding() {
        let policy = ExpertStreamingPolicy::quality_first(6);
        assert!(policy.preserve_source_encoding);
        assert_eq!(policy.gpu_slots_per_layer, 12);
        assert_eq!(policy.prefetch_per_layer, 6);
        assert!(!policy.allow_cpu_staging);
        assert!(!policy.allow_remote_sources);
    }

    #[test]
    fn host_cache_get_shares_bundle_allocation() {
        let mut cache = HostStagedExpertCache::with_limits(MemoryPoolLimits::new(2, 64));
        assert!(cache.insert(synthetic_bundle(0, 0, 12)));

        let first = cache.get(ExpertId::new(0, 0)).unwrap();
        let second = cache.get(ExpertId::new(0, 0)).unwrap();

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.total_bytes(), 12);
        assert_eq!(cache.hits(), 2);
    }

    #[test]
    fn host_cache_evicts_lru_until_byte_budget_is_satisfied() {
        let mut cache = HostStagedExpertCache::with_limits(MemoryPoolLimits::new(8, 12));
        assert!(cache.insert(synthetic_bundle(0, 0, 6)));
        assert!(cache.insert(synthetic_bundle(0, 1, 6)));
        assert!(cache.insert(synthetic_bundle(0, 2, 6)));

        assert!(!cache.contains(ExpertId::new(0, 0)));
        assert!(cache.contains(ExpertId::new(0, 1)));
        assert!(cache.contains(ExpertId::new(0, 2)));
        assert_eq!(cache.total_bytes(), 12);
        assert_eq!(
            cache.stats(),
            MemoryPoolStats {
                limits: MemoryPoolLimits::new(8, 12),
                entries_used: 2,
                bytes_used: 12,
                peak_bytes_used: 12,
                admissions: 3,
                evictions: 1,
                ..MemoryPoolStats::default()
            }
        );
    }

    #[test]
    fn host_cache_replacement_updates_bytes_without_counting_an_eviction() {
        let mut cache = HostStagedExpertCache::with_limits(MemoryPoolLimits::new(2, 20));
        assert!(cache.insert(synthetic_bundle(0, 0, 6)));
        assert!(cache.insert(synthetic_bundle(0, 1, 8)));
        assert!(cache.insert(synthetic_bundle(0, 0, 12)));

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.total_bytes(), 20);
        assert_eq!(cache.evictions(), 0);
        assert_eq!(cache.stats().peak_bytes_used, 20);
        assert_eq!(cache.get(ExpertId::new(0, 0)).unwrap().total_bytes(), 12);
        assert!(cache.contains(ExpertId::new(0, 1)));
    }

    #[test]
    fn host_cache_rejects_oversize_entry_without_disturbing_residents() {
        let mut cache = HostStagedExpertCache::with_limits(MemoryPoolLimits::new(2, 10));
        assert!(cache.insert(synthetic_bundle(0, 0, 8)));
        let resident = cache.get(ExpertId::new(0, 0)).unwrap();

        assert!(!cache.insert(synthetic_bundle(0, 0, 11)));

        let after_rejection = cache.get(ExpertId::new(0, 0)).unwrap();
        assert!(Arc::ptr_eq(&resident, &after_rejection));
        assert_eq!(cache.total_bytes(), 8);
        assert_eq!(cache.evictions(), 0);
        assert_eq!(cache.rejections(), 1);
    }

    #[test]
    fn host_cache_hit_promotes_entry_in_lru_order() {
        let mut cache = HostStagedExpertCache::with_limits(MemoryPoolLimits::new(2, 64));
        assert!(cache.insert(synthetic_bundle(0, 0, 4)));
        assert!(cache.insert(synthetic_bundle(0, 1, 4)));
        assert!(cache.get(ExpertId::new(0, 0)).is_some());
        assert!(cache.insert(synthetic_bundle(0, 2, 4)));

        assert!(cache.contains(ExpertId::new(0, 0)));
        assert!(!cache.contains(ExpertId::new(0, 1)));
        assert!(cache.contains(ExpertId::new(0, 2)));
        assert_eq!(cache.evictions(), 1);
    }

    #[test]
    fn rejected_async_cache_admission_preserves_one_shot_bundle_handoff() {
        let expert = ExpertId::new(0, 7);
        let mut loader = AsyncHostStagedExpertLoader::new(1);
        let mut cache = HostStagedExpertCache::with_limits(MemoryPoolLimits::disabled());
        let mut unretained = HashMap::new();

        assert!(loader.handle_result(
            AsyncHostStagedExpertResult::Loaded(Box::new(synthetic_bundle(0, 7, 12))),
            &mut cache,
            &mut unretained,
        ));

        assert!(cache.is_empty());
        assert_eq!(cache.rejections(), 1);
        assert_eq!(unretained.remove(&expert).unwrap().total_bytes(), 12);
        assert_eq!(loader.stats().completed, 1);
    }

    #[test]
    fn classifies_expert_residency_before_backend_materialization() {
        let source = ExpertLoadSource::LocalShard {
            path: PathBuf::from("model.safetensors"),
            offset: 0,
            bytes: 10,
        };
        let loads = vec![
            ExpertLoadRequest {
                expert: ExpertId::new(0, 1),
                load_source: source.clone(),
                reason: ExpertLoadReason::Selected,
            },
            ExpertLoadRequest {
                expert: ExpertId::new(0, 2),
                load_source: source.clone(),
                reason: ExpertLoadReason::Selected,
            },
            ExpertLoadRequest {
                expert: ExpertId::new(0, 3),
                load_source: source.clone(),
                reason: ExpertLoadReason::Selected,
            },
            ExpertLoadRequest {
                expert: ExpertId::new(0, 8),
                load_source: source.clone(),
                reason: ExpertLoadReason::Selected,
            },
            ExpertLoadRequest {
                expert: ExpertId::new(0, 4),
                load_source: source.clone(),
                reason: ExpertLoadReason::Prefetch,
            },
            ExpertLoadRequest {
                expert: ExpertId::new(0, 5),
                load_source: source.clone(),
                reason: ExpertLoadReason::Prefetch,
            },
            ExpertLoadRequest {
                expert: ExpertId::new(0, 6),
                load_source: source.clone(),
                reason: ExpertLoadReason::Prefetch,
            },
            ExpertLoadRequest {
                expert: ExpertId::new(0, 9),
                load_source: source.clone(),
                reason: ExpertLoadReason::Prefetch,
            },
            ExpertLoadRequest {
                expert: ExpertId::new(0, 7),
                load_source: source,
                reason: ExpertLoadReason::Prefetch,
            },
        ];
        let plan = classify_expert_residency(
            &loads,
            |expert| matches!(expert.expert, 1 | 4),
            |expert| matches!(expert.expert, 8 | 9),
            |expert| matches!(expert.expert, 2 | 5),
            |expert| matches!(expert.expert, 3 | 6),
        );

        assert_eq!(plan.selected_resident, vec![ExpertId::new(0, 1)]);
        assert_eq!(
            plan.selected_host_staged
                .iter()
                .map(|load| load.expert)
                .collect::<Vec<_>>(),
            vec![ExpertId::new(0, 2)]
        );
        assert_eq!(plan.selected_materializing, vec![ExpertId::new(0, 8)]);
        assert_eq!(
            plan.selected_in_flight
                .iter()
                .map(|load| load.expert)
                .collect::<Vec<_>>(),
            vec![ExpertId::new(0, 3)]
        );
        assert!(plan.selected_cold.is_empty());
        assert_eq!(plan.prefetch_materializing, vec![ExpertId::new(0, 9)]);
        assert_eq!(plan.prefetch_host_staged, vec![ExpertId::new(0, 5)]);
        assert_eq!(plan.prefetch_in_flight, vec![ExpertId::new(0, 6)]);
        assert_eq!(
            plan.prefetch_cold
                .iter()
                .map(|load| load.expert)
                .collect::<Vec<_>>(),
            vec![ExpertId::new(0, 7)]
        );
        assert_eq!(plan.selected_miss_count(), 3);
        assert_eq!(plan.prefetch_enqueue_count(), 1);
        assert_eq!(plan.prefetch_skipped_cached_or_inflight_count(), 3);
    }

    #[test]
    fn loads_selected_experts_from_local_shards_without_cpu_staging() {
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy::quality_first(2));
        for expert in 0..4 {
            planner.register_load_source(
                ExpertId::new(0, expert),
                ExpertLoadSource::LocalShard {
                    path: PathBuf::from(format!("shard-{expert}.safetensors")),
                    offset: expert as u64 * 1024,
                    bytes: 1024,
                },
            );
        }

        let step = planner.plan_layer_step(0, &[1, 3], &[]).unwrap();
        assert_eq!(
            step.selected,
            vec![ExpertId::new(0, 1), ExpertId::new(0, 3)]
        );
        assert_eq!(step.loads.len(), 2);
        assert!(
            step.loads
                .iter()
                .all(|load| load.reason == ExpertLoadReason::Selected)
        );
        planner.commit_step(&step).unwrap();
        assert_eq!(
            planner.location(ExpertId::new(0, 1)),
            Some(ExpertStorageTier::Gpu)
        );
        assert_eq!(
            planner.location(ExpertId::new(0, 3)),
            Some(ExpertStorageTier::Gpu)
        );
    }

    #[test]
    fn prefetch_uses_remaining_slots_but_never_duplicates_selected() {
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy {
            gpu_slots_per_layer: 3,
            prefetch_per_layer: 2,
            preserve_source_encoding: true,
            allow_cpu_staging: false,
            allow_remote_sources: false,
        });
        for expert in 0..5 {
            planner.register_load_source(
                ExpertId::new(0, expert),
                ExpertLoadSource::LocalShard {
                    path: PathBuf::from("model.safetensors"),
                    offset: 0,
                    bytes: 10,
                },
            );
        }

        let step = planner.plan_layer_step(0, &[2, 2], &[2, 3, 4]).unwrap();
        assert_eq!(step.selected, vec![ExpertId::new(0, 2)]);
        assert_eq!(
            step.prefetched,
            vec![ExpertId::new(0, 3), ExpertId::new(0, 4)]
        );
        assert_eq!(step.loads.len(), 3);
    }

    #[test]
    fn commit_step_loaded_keeps_unmaterialized_prefetches_non_resident() {
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy {
            gpu_slots_per_layer: 3,
            prefetch_per_layer: 2,
            preserve_source_encoding: true,
            allow_cpu_staging: false,
            allow_remote_sources: false,
        });
        for expert in 0..5 {
            planner.register_load_source(
                ExpertId::new(0, expert),
                ExpertLoadSource::LocalShard {
                    path: PathBuf::from("model.safetensors"),
                    offset: 0,
                    bytes: 10,
                },
            );
        }

        let step = planner.plan_layer_step(0, &[2], &[3, 4]).unwrap();
        planner
            .commit_step_loaded(&step, [ExpertId::new(0, 2)])
            .unwrap();

        assert_eq!(
            planner.location(ExpertId::new(0, 2)),
            Some(ExpertStorageTier::Gpu)
        );
        assert_ne!(
            planner.location(ExpertId::new(0, 3)),
            Some(ExpertStorageTier::Gpu)
        );
        assert_ne!(
            planner.location(ExpertId::new(0, 4)),
            Some(ExpertStorageTier::Gpu)
        );
    }

    #[test]
    fn retains_recent_resident_experts_when_slots_are_available() {
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy {
            gpu_slots_per_layer: 2,
            prefetch_per_layer: 0,
            preserve_source_encoding: true,
            allow_cpu_staging: false,
            allow_remote_sources: false,
        });
        for expert in 0..4 {
            planner.register_load_source(
                ExpertId::new(0, expert),
                ExpertLoadSource::LocalShard {
                    path: PathBuf::from("model.safetensors"),
                    offset: 0,
                    bytes: 10,
                },
            );
        }

        let first = planner.plan_layer_step(0, &[0], &[]).unwrap();
        planner.commit_step(&first).unwrap();
        let second = planner.plan_layer_step(0, &[2], &[]).unwrap();

        assert_eq!(second.loads.len(), 1);
        assert!(second.evictions.is_empty());
        planner.commit_step(&second).unwrap();
        assert_eq!(
            planner.resident_experts(0),
            vec![ExpertId::new(0, 0), ExpertId::new(0, 2)]
        );
    }

    #[test]
    fn hotset_reports_observed_experts_by_frequency() {
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy {
            gpu_slots_per_layer: 3,
            prefetch_per_layer: 0,
            preserve_source_encoding: true,
            allow_cpu_staging: false,
            allow_remote_sources: false,
        });
        for expert in 0..4 {
            planner.register_load_source(
                ExpertId::new(0, expert),
                ExpertLoadSource::LocalShard {
                    path: PathBuf::from("model.safetensors"),
                    offset: 0,
                    bytes: 10,
                },
            );
        }

        let first = planner.plan_layer_step(0, &[1, 2], &[]).unwrap();
        planner.commit_step(&first).unwrap();
        let second = planner.plan_layer_step(0, &[1], &[]).unwrap();
        planner.commit_step(&second).unwrap();

        assert_eq!(planner.hot_experts(0, 3), vec![1, 2]);
        assert!(planner.hot_experts(1, 3).is_empty());
    }

    #[test]
    fn eviction_keeps_hot_resident_experts_before_cold_recent_ones() {
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy {
            gpu_slots_per_layer: 2,
            prefetch_per_layer: 0,
            preserve_source_encoding: true,
            allow_cpu_staging: false,
            allow_remote_sources: false,
        });
        for expert in 0..4 {
            planner.register_load_source(
                ExpertId::new(0, expert),
                ExpertLoadSource::LocalShard {
                    path: PathBuf::from("model.safetensors"),
                    offset: 0,
                    bytes: 10,
                },
            );
        }

        for selected in [[0], [0], [1]] {
            let step = planner.plan_layer_step(0, &selected, &[]).unwrap();
            planner.commit_step(&step).unwrap();
        }
        let step = planner.plan_layer_step(0, &[2], &[]).unwrap();

        assert_eq!(step.loads.len(), 1);
        assert_eq!(step.evictions.len(), 1);
        assert_eq!(step.evictions[0].expert, ExpertId::new(0, 1));
    }

    #[test]
    fn evicts_non_target_gpu_experts_to_artifact_tier() {
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy {
            gpu_slots_per_layer: 2,
            prefetch_per_layer: 0,
            preserve_source_encoding: true,
            allow_cpu_staging: false,
            allow_remote_sources: false,
        });
        for expert in 0..4 {
            planner.register_load_source(
                ExpertId::new(0, expert),
                ExpertLoadSource::LocalShard {
                    path: PathBuf::from("model.safetensors"),
                    offset: 0,
                    bytes: 10,
                },
            );
        }
        planner
            .mark_resident(ExpertId::new(0, 0), ExpertStorageTier::Gpu)
            .unwrap();
        planner
            .mark_resident(ExpertId::new(0, 1), ExpertStorageTier::Gpu)
            .unwrap();

        let step = planner.plan_layer_step(0, &[2, 3], &[]).unwrap();
        assert_eq!(step.loads.len(), 2);
        assert_eq!(step.evictions.len(), 2);
        assert!(
            step.evictions
                .iter()
                .all(|evict| evict.target == ExpertStorageTier::LocalStorage)
        );
        planner.commit_step(&step).unwrap();
        assert_eq!(
            planner.resident_experts(0),
            vec![ExpertId::new(0, 2), ExpertId::new(0, 3)]
        );
    }

    #[test]
    fn rejects_remote_sources_until_policy_allows_them() {
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy::quality_first(1));
        planner.register_load_source(
            ExpertId::new(0, 7),
            ExpertLoadSource::Remote {
                uri: "http://lan-node/experts/l0e7".into(),
                offset: 0,
                bytes: 2048,
            },
        );
        let err = planner.plan_layer_step(0, &[7], &[]).unwrap_err();
        assert!(err.to_string().contains("allow_remote_sources=true"));
    }

    #[test]
    fn supports_remote_sources_for_no_local_storage_mode() {
        let mut planner =
            ExpertStreamingPlanner::new(ExpertStreamingPolicy::quality_first_remote(1));
        planner.register_load_source(
            ExpertId::new(0, 7),
            ExpertLoadSource::Remote {
                uri: "http://lan-node/experts/l0e7".into(),
                offset: 0,
                bytes: 2048,
            },
        );
        let step = planner.plan_layer_step(0, &[7], &[]).unwrap();
        assert_eq!(step.loads.len(), 1);
        assert_eq!(step.loads[0].load_source.tier(), ExpertStorageTier::Remote);
    }

    #[test]
    fn errors_when_selected_experts_exceed_concurrent_slots() {
        let mut planner = ExpertStreamingPlanner::new(ExpertStreamingPolicy {
            gpu_slots_per_layer: 1,
            prefetch_per_layer: 0,
            preserve_source_encoding: true,
            allow_cpu_staging: false,
            allow_remote_sources: false,
        });
        for expert in 0..2 {
            planner.register_load_source(
                ExpertId::new(0, expert),
                ExpertLoadSource::LocalShard {
                    path: PathBuf::from("model.safetensors"),
                    offset: 0,
                    bytes: 10,
                },
            );
        }
        let err = planner.plan_layer_step(0, &[0, 1], &[]).unwrap_err();
        assert!(
            err.to_string()
                .contains("selected experts must be available")
        );
    }

    fn synthetic_bundle(layer: usize, expert: usize, bytes: usize) -> ExpertComputeBundle {
        let expert = ExpertId::new(layer, expert);
        let linear = |matrix, bytes| ExpertLinearPayload {
            matrix,
            weight: ExpertTensorPayload {
                slice: ExpertTensorSlice {
                    key: ExpertTensorKey { expert, matrix },
                    component: ExpertTensorComponent::Weight,
                    path: PathBuf::from("synthetic.safetensors"),
                    offset: 0,
                    bytes: bytes as u64,
                    dtype: "opaque".into(),
                    shape: vec![bytes],
                },
                bytes: vec![1; bytes],
            },
            scale: None,
            format: ExpertLinearFormat::Opaque,
        };
        ExpertComputeBundle {
            expert,
            gate: linear(ExpertMatrixKind::Gate, bytes),
            up: linear(ExpertMatrixKind::Up, 0),
            down: linear(ExpertMatrixKind::Down, 0),
        }
    }

    pub(super) fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nonce}"))
    }
}
