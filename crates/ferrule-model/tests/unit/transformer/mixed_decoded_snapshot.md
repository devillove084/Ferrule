# Read-only model decoded-cache snapshot — runtime API handoff

## Exact API

```rust
use ferrule_model::transformer::{
    CudaStandardDecoderOperators,
    MixedDecodedCacheSnapshot,
    MixedDecodedCachePhase,
    MixedDecodedCacheActivity,
};

impl CudaStandardDecoderOperators {
    pub fn mixed_decoded_cache_snapshot(&self)
        -> Option<MixedDecodedCacheSnapshot>;
}
```

The public value types are `Debug + Clone + Copy + PartialEq + Eq + Serialize`.
`MixedDecodedCacheActivity` also implements `Default`. Authoritative field
names/types/docs: `src/transformer/standard/decoded_snapshot.rs`.
Phase serializes as `preparing`, `executing`, `idle`, or `quarantined`.
The getter takes only `&self`, emits no log, and is not a completion receipt.

## Fields and scope

All byte/grant fields and container counts are `usize`; lifetime counters and
image generation are `u64`. No field includes original FP8 weights, providers,
activation/state/KV buffers, unrelated allocator roots, or OS/NVML memory.

| Fields | Meaning |
| --- | --- |
| `image_generation`, `phase`, `published`, `quarantined` | Current image/owner metadata flags; not a CUDA proof |
| `capacity_grants` | Cache's existing hard physical ceiling |
| `pin_target_grants` | Entire admitted pin target |
| `pinned_grants` | Allocated fixed backing; a preparing decode need not be fenced yet |
| `unallocated_pin_grants` | Target minus allocated pins; SUBSET of pending reservation |
| `active_routed_grants` | Routed roots under keys, including borrowed/not-yet-Idle leases |
| `pooled_grants` | Physically retained completed reusable backing |
| `pending_reserved_grants` | Backing reservation without roots: future pins, allocation tickets, retained failed-allocation grants |
| `pending_allocation_count` | Outstanding allocation metadata tickets |
| `pending_lease_count`, `borrowed_key_count` | Outstanding logical routed leases / all borrowed cache keys |
| `segment_slack_bytes` | Existing conservative settled cache segment-slack reserve |
| `pending_segment_slack_bytes` | Existing in-flight worst-case slack reservation |
| `physical_current_bytes`, `physical_peak_bytes` | Existing cache charge and reservation high-water, NOT measured GPU peak |
| `driver_allocation_growth_bytes` | Existing cumulative owner-allocator growth around cache allocations; not added to current again |
| `unknown_reserved_bytes` | Cache charge retained under observed quarantine |
| `backing_reuses`, `native_fallbacks`, `evictions` | Existing whole-image counters, available with profiling off |
| `rank_performance_enabled` | Requested first rankperf switch |
| `rank_performance_active` | Actual active rankperf invocation, admitted by both existing gates |
| `profiled_activity` | Optional cumulative observed invocation counts, defined below |

Exact accounting identity, including during preparation/execution:

```
physical_current_bytes = pinned_grants + active_routed_grants + pooled_grants
                       + pending_reserved_grants
                       + segment_slack_bytes + pending_segment_slack_bytes
pin_target_grants     = pinned_grants + unallocated_pin_grants
```

Do not add `pin_target_grants`, `unallocated_pin_grants`, or
`driver_allocation_growth_bytes` to the first identity. After a healthy failed
allocation, ticket/lease counts may be zero while `pending_reserved_grants`
remains positive: this is retained reservation, not unreported free backing.
Peak includes earlier worst-case allocation tickets and is not reset per stage.

## Preparation / execute / after-complete protocol

- `prepare_first`, before cache custody exists: getter returns **None**. This
  means absent custody, not measured zero; it can also mean cache disabled or
  a healthy preparation rollback. Caller knows which stage/config applies.
- After custody install but before preparation returns: `Preparing`; pin target
  is already reserved, while allocated pins and pending counts evolve.
- After healthy preparation returns: published + `Idle`; pins are allocated,
  and borrowed/pending lease sets are clear. Snapshot creates no new proof.
- Inside an existing synchronous operation: `Executing` while its depth is
  nonzero. Active backing can still have borrowed/pending consumers.
- Between synchronous operations within a profiled invocation, phase can be
  `Idle` while `rank_performance_active == true`; these are different scopes.
- After existing outer healthy completion: `Idle`, no outstanding leases or
  borrowed keys. Cache bytes are retained (active/pool), not released by the
  getter. Failed-allocation reservations may remain by existing policy.
- Unknown dominates every phase as `Quarantined`; snapshot does not fence,
  recover, release, reset or reuse anything. Capture shutdown snapshot BEFORE
  dropping the owner. Getter cannot observe a dropped image.

Runtime may label snapshots `prepare_first`, `ready`, `execute_end`, `shutdown`
or its diagnostic stage without passing a label into model or pretending that
a label proves completion. No model per-call logging is introduced.

## Optional activity counters — never substitute zero for disabled

```rust
pub struct MixedDecodedCacheActivity {
    pub profiled_invocations: u64,
    pub hits: u64,
    pub misses: u64,
    pub decoded_bytes: u64,
    pub evictions: u64,
}
```

- `profiled_activity == None`: no completed OR active observed invocation.
  This includes preparation and the case where the setter was enabled but the
  tracing gate never enabled rankperf. Do not serialize None as zero counters.
- `Some`: existing finalized rankperf counters plus the currently active
  invocation's counters if `rank_performance_active`. Repeated reads do not
  accumulate or double-count that active invocation.
- `profiled_invocations` counts those observed invocation scopes (active one
  included), not successful GPU completions or the total model invocation count.
- Finalized observations survive disabling rankperf and are available at
  shutdown. They never claim coverage of preparation or disabled intervals.
- Existing observer semantics are unchanged: misses include successful cache
  decode returns and pre-submission native fallbacks; decoded bytes count BF16
  target bytes of successful decode API returns, not GPU-completed work.
  Errored/Unknown decode submissions may not appear. Pair bytes are full 2NK.
- Activity counters saturate; snapshot's unconditional lifetime `evictions`
  can differ from profiled activity evictions because their coverage differs.

Both existing gates remain unchanged/default off:
`set_mixed_rank_performance(true)` AND an enabled `ferrule_performance=debug`
tracing target. This getter does not parse RUST_LOG or enable either switch.
Runtime owns any explicit RUST_LOG opt-in policy. No runtime/composition files
were edited for this API.

## Cost / production invariants

- Snapshot: fixed scalar copies/subtractions plus O(1) container `.len()` and,
  only if rankperf is active, one immutable borrow of its existing four counters.
  No CUDA or allocator query, synchronization, payload/source read, clock,
  allocation, map iteration, or log emission.
- Fixed-pin allocations add one observation-only integer increment when their
  existing root is installed (or a matching root is reused); not on hits.
- Existing per-hit/per-decode observer code is untouched. At an already-profiled
  invocation end, archive four existing counters and one invocation count into
  the cache image. Profiling off adds no per-call counter updates.
- New observation fields are NEVER read by admission, victim selection, decode,
  buffer reuse, arithmetic, geometry, fallback decisions or completion logic.
- No CUDA kernel, buffer size, cache policy, provider, fence or source-identity
  behavior changes; this is observability only.

## Tests

`cargo test -p ferrule-model --lib mixed_decoded -- --test-threads=1` includes
snapshot invariants for preparation/active/after-complete, pools, retained failed
reservations, Unknown, immutable repeated reads, active/finalized profile
aggregation, disabled-vs-zero semantics, JSON null and saturating counters.
With CUDA enabled, a host-only type test verifies the public shared-borrow API.
CUDA-hidden Clippy checks all model libraries/tests; no GPU validation is
required or claimed by these CPU snapshot tests.

Validation for this observability-only change: 19 default-feature CPU
cache/snapshot tests passed (4 new snapshot tests), existing timing 8 passed /
1 calibration ignored, upload-proof 6 passed; CUDA-hidden model Clippy
`--lib --tests -D warnings` passed, including compilation of the public API
signature test. No new GPU run or numeric revalidation was performed here.
