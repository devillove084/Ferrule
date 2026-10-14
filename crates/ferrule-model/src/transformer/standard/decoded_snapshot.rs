//! Read-only decoded-cache accounting, not a CUDA completion receipt.

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MixedDecodedCachePhase {
    Preparing,
    Executing,
    /// Published image outside a synchronous operation; not a new fence proof.
    Idle,
    Quarantined,
}

/// Cumulative observations ONLY from rankperf-enabled invocations. All counters
/// saturate. Preparation and unprofiled execution are deliberately excluded.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct MixedDecodedCacheActivity {
    /// Includes the current invocation iff snapshot.rank_performance_active.
    pub profiled_invocations: u64,
    pub hits: u64,
    /// Existing rankperf semantics: successful decode misses + native fallbacks;
    /// failed/Unknown decode submissions are not successful observations.
    pub misses: u64,
    /// BF16 bytes of successfully returned decode calls, NOT GPU-completed bytes.
    pub decoded_bytes: u64,
    pub evictions: u64,
}
impl MixedDecodedCacheActivity {
    pub(super) fn with_invocation(mut self, counters: [u64; 4]) -> Self {
        self.profiled_invocations = self.profiled_invocations.saturating_add(1);
        self.hits = self.hits.saturating_add(counters[0]);
        self.misses = self.misses.saturating_add(counters[1]);
        self.decoded_bytes = self.decoded_bytes.saturating_add(counters[2]);
        self.evictions = self.evictions.saturating_add(counters[3]);
        self
    }
}

/// Constant-time, allocation-free COPY of the image's host-side ledger.
/// Byte fields describe cache-only backing grants/reservations, not NVML usage,
/// OS free memory, provider workspace, or a new last-consumer proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct MixedDecodedCacheSnapshot {
    pub image_generation: u64,
    pub phase: MixedDecodedCachePhase,
    pub published: bool,
    pub quarantined: bool,
    pub capacity_grants: usize,
    /// Installed physical root only; a subset of physical_current_bytes, NOT
    /// an additional charge on top of the logical slot fields below.
    pub arena_backing_grants: usize,
    /// Entire admitted fixed-pin target, including not-yet-created views.
    pub pin_target_grants: usize,
    /// Logical fixed slots with views, including not-yet-fenced submissions.
    pub pinned_grants: usize,
    /// Subset of pending_reserved_grants, not an additional charge.
    pub unallocated_pin_grants: usize,
    /// Allocated routed backing under keys, including borrowed/pending leases.
    pub active_routed_grants: usize,
    /// Completed empty slots retained for reuse inside the root.
    pub pooled_grants: usize,
    /// Never-bound routed slots with views, not extra physical backing.
    pub free_slot_grants: usize,
    /// Unassigned root tail. Included exactly once in the root reservation.
    pub unused_tail_bytes: usize,
    /// Padding inside planned slots, a subset of slot grants, never added again.
    pub slot_padding_bytes: usize,
    /// Reserved bytes without constructed views (before/failed root install or
    /// partial view construction). Not necessarily bytes without a root.
    pub pending_reserved_grants: usize,
    /// These counts observe metadata containers in O(1); zero allocation tickets
    /// does not erase a prior failed allocation's retained reservation.
    pub pending_allocation_count: usize,
    pub pending_lease_count: usize,
    pub borrowed_key_count: usize,
    /// Legacy segmented-allocation fields: always zero for the single arena.
    pub segment_slack_bytes: usize,
    pub pending_segment_slack_bytes: usize,
    /// One full root reservation, retained on failure/Unknown. Equivalently:
    /// pinned + active_routed + pooled + free_slot + unused_tail + pending_reserved.
    /// Do NOT add arena_backing_grants or slot_padding_bytes to that partition.
    pub physical_current_bytes: usize,
    /// Existing physical ledger high-water, including worst-case reservations.
    pub physical_peak_bytes: usize,
    /// Observed growth around cache allocations; NOT added again to current.
    pub driver_allocation_growth_bytes: usize,
    pub unknown_reserved_bytes: usize,
    /// Lifetime counters already maintained even with rankperf disabled.
    pub backing_reuses: u64,
    pub native_fallbacks: u64,
    pub evictions: u64,
    /// First profiling switch requested by set_mixed_rank_performance.
    pub rank_performance_enabled: bool,
    /// An actual rankperf invocation is open (both existing gates admitted it).
    pub rank_performance_active: bool,
    /// None = no recorded or active profiled invocation, NOT measured zero.
    /// Some = finalized profiled invocations plus the current one, if active.
    /// Survives disabling profiling, but never pretends to cover disabled gaps.
    pub profiled_activity: Option<MixedDecodedCacheActivity>,
}
