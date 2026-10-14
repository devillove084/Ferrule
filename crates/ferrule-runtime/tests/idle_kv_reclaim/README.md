# Idle/suspended KV reclaim

## Supported benefit and limits

`KvExecutionPolicy::IdleReclaim(IdleKvReclaimConfig)` explicitly selects the
local dense F32 CUDA chunked KV runner. The default remains the fixed pool.
The supported benefit is **physical VRAM release while a session is idle or
suspended**, followed by exact reload before its next execution. This is not
active CPU attention, constant-memory attention, or evidence of a smaller active
working set. No model weights are moved to CPU.

The public serving factory currently supports this policy for full-depth local
dense Qwen3 F32 all-GQA, without numeric/GDN/MoE/parallel overrides. The separately
validated typed BF16 backend and ongoing BF16 model work do not extend this
F32 policy's supported configurations.

## Public programmatic construction

Use the public factory, not the private `model_factory::resident_f32_bound::build_resident_f32_bound` test seam:

```rust
use ferrule_model::AutoConfig;
use ferrule_runtime::engine::BoxedSessionInferenceEngine;
use ferrule_runtime::{IdleKvReclaimConfig, KvExecutionPolicy};
use ferrule_composition::{
    BackendSelection, ModelFactoryOptions, ResidentModelPlanner, ServingModelOptions,
};

fn build_idle_engine(
    model: &AutoConfig,
    options: ModelFactoryOptions,
    budget: IdleKvReclaimConfig,
) -> ferrule_runtime::Result<BoxedSessionInferenceEngine> {
    ResidentModelPlanner::new()
        .prepare_serving(
            model,
            BackendSelection::Cuda,
            None,
            options,
            ServingModelOptions {
                kv_execution_policy: KvExecutionPolicy::IdleReclaim(budget),
                ..Default::default()
            },
        )?
        .build()
}
```

Supply the model metadata, scheduler/context options and explicit budget for the
actual deployment. There is no automatic/default budget or CLI reclaim flag.
The build constructs chunked storage and installs the matching policy before
admission. Calling `configure_kv_execution(IdleReclaim(...))` on an already-built
fixed runner is not a storage conversion and is rejected. Switching a chunked
runner to `Resident` or `HostKvDecode` is also rejected.

For direct model composition, the public constructor is
`GenericDecoderRunner::<HybridCudaDecoder>::hybrid_cuda_with_idle_kv_reclaim`.
Use `HybridCudaMemoryEstimate::for_resources_with_idle_kv_reclaim` for admission,
then `ferrule_runtime::engine::build_resident_engine` with the same physical
capacity. These are public methods, not fixture-only hooks.

## Public session lifecycle

The returned `BoxedSessionInferenceEngine` implements
`ferrule_runtime::engine::SessionInferenceEngine`:

- `retain_session(session_id)`: opt into retaining a session across completed
  requests. Continue stepping and consume the request terminal before treating
  it as idle; receiving a token is not a quiescence proof.
- `suspend_session_kv(session_id)`: remove a quiescent active session from
  scheduling, or park an idle retained session. Snapshot all planes of exclusively
  owned pages and try allocator trim. Repeating suspension is rejected.
- `trim_idle_kv()`: nonblocking maintenance after retirement events become ready;
  no live transaction custody may be bypassed. A successful call need not free
  any driver memory. A trim failure after successful suspension does not undo
  suspension or authorize replaying it.
- `restore_session_kv(session_id)`: complete physical H2D and logical restore
  before execution can resume. A consumed suspension cannot be restored twice.
- `retained_session_position(session_id)` and `observability_snapshot()`: inspect
  the frontier and physical gauges without acknowledging any work.

The same public methods are available on `ResidentTopKDriver` and
`ferrule_runtime::engine::LocalSessionInferenceEngine`. Explicitly restore before
submitting more work to a suspended session; no implicit resident fallback is
performed. Other sessions may continue while one is suspended.

All paths use the existing `KvPageManager`, one-use opaque suspension
registration and transaction Entry. Logical page IDs/frontier/COW relationships
are retained; restore advances the sequence incarnation, so old committed-read
witnesses do not revive. Shared or snapshot-pinned pages remain resident. Active
transaction, COW, prefix-reader and committed-reader custody blocks preemption.
A failed or canceled operation with Unknown completion keeps the physical and
logical credits; a later successful CUDA event does not clear Unknown.

## Staging and capacity audit

The active working set still needs device residency. Entry staging **copies all
declared pages**, not just this query's packed block table: new pages, protected
reads and COW sources, with the provisional slot replacing an original writable
slot in the compact view. Authoritative originals remain charged separately.

The production guards remain unchanged:

1. `prepare_with_submission` calls
   `admit(plan.original.len(), staging_page_limit, entries.len())` **before**
   reserving physical slots. It rejects an oversized declaration or another live
   transaction; one transaction bounds aggregate Entry staging per pool.
2. `gather_chunked` uses `compact_slots` with the same limit and copies every
   declared slot/plane. It rejects aliasing/overflow rather than truncating or
   silently streaming the working set.
3. Construction checks `0 < staging_page_limit <= physical_slots`, and
   `IdleKvReclaimConfig::plan` checks all budget domains with overflow detection.

Staging can therefore fill the entire declared working set. Choose its limit for
the largest admitted packed transaction, including COW sources; idle offload does
not relax that limit and no active working-set reduction is claimed.

`IdleKvReclaimConfig` admits separate domains:

- `device_bytes`: dedicated per-page allocator grants for the physical slot
  capacity (including original/provisional shadows), plus bounded Entry staging
  grants, with each allocation's rounding accounted for;
- `host_bytes`: complete all-plane snapshots up to the configured page capacity;
- `pinned_bytes`: full-batch all-plane DMA frames, charged alongside snapshots;
- `staging_page_limit`: the declared-page bound described above.

These KV admission amounts are not context-wide driver reservation limits.
Weights, operator scratch, cached allocator segments and pending retirements are
separate from live KV payload. Runtime
`observability_snapshot().kv_cache.unwrap().idle_physical` exposes the KV
admission and live payload/staging/host/pinned gauges separately from
context-wide allocator live/reserved/pending bytes and cumulative
`driver_freed_bytes`. Legacy logical page counts or configured payload bytes are
not actual driver residency or savings.

## Owner GPU acceptance and fixture-only correction

Recorded source of evidence:
`target/validation/qwen35-35b-current/idle-bf16-owner/REPORT.md` and
`runtime-unknown-fixture-fix.diff` in the same directory. The independent owner
ran CUDA 12.4 on physical GPU1; this synchronization does not rerun GPU work.

All three intended runtime GPU fixtures passed. Full-logit comparisons reported
maximum absolute error `0.0`. Factory telemetry (bytes):

| Boundary | Live KV payload | Host snapshot | Context cumulative driver freed |
| --- | ---: | ---: | ---: |
| Before offload | 4096 | 0 | 0 |
| One session suspended | 2048 | 2048 | 30720 |
| After maintenance trim | 2048 | 2048 | 43008 |
| After restore | 4096 | 0 | 43008 |

Driver-free counters include earlier dead allocations in the same context;
`30720`/`43008` must **not** be attributed to the single 2048-byte page payload.
This proves idle release/reload, not active working-set savings or a whole-BF16
model/profile result.

The original Unknown fixture failed at construction, before the DMA failpoint:
`PageLimit(32)` was capped at 16 context pages, while its runner owned 32 slots.
The live fixture now matches the owner correction exactly:
`TransactionCapacity { page_bytes: schema.checked_page_bytes().unwrap() as u64,
budget_bytes: None }`, with an assertion for 16 context + 16 shadow slots.
No production capacity guard or budget was weakened. The corrected owner run
reached the real failpoint and retained `unknown=true`, device payload 256 bytes,
pinned 256 bytes and driver freed 0 bytes. Later completion did not reset it.

Exact ignored fixture names under
`model_factory::qwen35::tests::idle_reclaim_tests::`:

- `engine_two_sessions_idle_offload_one_restore_and_continue_matches_fixed`
- `runner::full_logits_two_sequences_reload_and_cow_match_fixed_pool`
- `runner::actual_engine_cancel_after_unknown_dma_retains_physical_and_logical_credits`

For no-GPU validation, source
`target/validation/qwen35-35b-current/toolchain/env-cuda124.sh`, **then** override
`CUDA_VISIBLE_DEVICES=` (the script initially selects a GPU). Run CPU protocol
and suspension tests, CUDA compilation with `--no-run`, and Clippy; do not use
`--ignored` while the GPU owner is busy. Every command in this synchronization
is bounded to 600000 ms.

### Live-worktree synchronization checks

The owner fixture correction was already present and was retained (only its
call formatting was normalized). This synchronization changes tests/docs, not
production admission, budgets, capacity guards, or BF16 model/operator code.
Added CPU regressions cover the exact 16-context/16-shadow assembly and the
all-declared-pages staging bound, including pages omitted from a packed batch.

The backend physical CPU suite passed **5/5**; CUDA backend/model library checks
and Clippy with `-D warnings` passed, as did formatting and diff checks. Current
runtime idle/suspension tests cannot build because
`src/parallel/process/ipc.rs` compares `str` in a const (`E0658`). The attempted
`cargo clippy --workspace --all-targets --features cuda -- -D warnings` is blocked
by `Cell<Vec<_>>::get()` in backend `src/cuda/transport_wait.rs` tests (`E0599`).
These unrelated owner files were not modified. This blocked live-worktree
validation is separate from the successful independent GPU acceptance above.
