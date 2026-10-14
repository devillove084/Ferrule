# Dedicated uninitialized Mixed decoded arena handoff

## Custody and readiness contract

The model calls `allocate_bf16_arena_uninitialized(arena_grants / 2,
arena_grants)` with the full admitted grant. On success it immediately calls
`install_arena(RefCell::new(root), arena_grants, slice)`. No allocator observation,
CUDA submission, initialization or synchronization occurs between return and
custody. Core stores the root before invoking the first fallible slice closure.

The successful `driver_growth` argument is **not an estimate**: the backend's
`allocate_dedicated` skips reuse, allocates a new driver segment of exactly the
rounded grant, and this plan already requires a grant-aligned root. This relies
on the dedicated allocator contract; do not replace it with pooled allocation
while retaining this accounting. No split-install core API was needed.

Before allocation, metrics provide an attribution baseline for the error path.
Only when allocation returns `Err` does production query the after-metrics and
record actual growth with `arena_allocation_failed`; no returned root exists in
that branch. On success, after-metrics are test-only and occur after installation.

The arena is never zeroed as a whole. Fixed slots are fully decoded before ready;
all fixed preparation and existing healthy completion proofs precede publication.
Free routed slots remain uninitialized and cannot be consumed. Acquisition of an
already borrowed, not-ready key fails before a GPU consumer can be submitted.
Single decode writes the complete slot; pair decode writes both ordered halves.
Successful same-stream decode permits `mark_ready`, not premature Idle publication.

View installation error/Unknown/panic retains the entire root and reservation.
First-decode Unknown/panic is terminal and also retains the entire root. A normal
first-decode failure retains custody until the existing completion proves work
finished; only then may healthy preparation rollback release it. No new fences.

## Assertions and small gates

Test-only metrics bracket **only arena allocation plus view installation**, not
GemmEx storage or source uploads. Required deltas: one allocation request, one
huge allocation, one driver allocation, exactly the full grant in granted bytes
and driver allocation bytes, and zero allocator reuse. Kernel, H2D, D2H and
stream-sync counters must not increase across that boundary. Backend structural
checks additionally prohibit the uncounted driver memset path.

Existing `decode_completed` cold/hot/reload checks require zero new allocation
requests, granted bytes and driver allocation bytes after preparation. This is a
decoded-cache boundary assertion, not a claim that model activations never allocate.
Exact-class quotas, full-root charge, tail/padding accounting and native fallback
remain unchanged; legacy segment-slack fields stay zero.

All GPU selectors below have the prefix:

`transformer::standard::cuda::hybrid_rank::mixed::decoded_gpu_tests::`

New small gates (all ignored, sole GPU owner required):

- `decoded_uninitialized_arena_view_faults_retain_whole_root`
- `decoded_first_fixed_failure_retained_until_completion_not_ready`
- `decoded_uninitialized_slots_reject_consumers_until_ready`

Also rerun existing fixed/routed completion and numerical gates:

- `decoded_pair_single_cold_hot_fit_exact_and_no_cache_dma`
- `decoded_fixed_pins_prepare_once_and_shared_stays_separate`
- `decoded_arena_fit_working_set_hot_zero_eviction_decode_fallback`
- `decoded_arena_true_class_full_lru_and_size_class_fairness`
- `decoded_physical_full_native_fallback_before_submit_exact`
- `decoded_pair_unknown_and_panic_retain_physical_backing`
- `decoded_fixed_pin_unknown_and_panic_never_publish_arena`
- `decoded_arena_alias_retains_whole_root`

GPU execution requires the existing owner authorization and
`FERRULE_RUN_MIXED_RANK_GPU=1`, `CUBLAS_WORKSPACE_CONFIG=:0:0`. The numerical gates
use lanes 0 and 1. Run selectors with `--exact --ignored --test-threads=1` in the
owner's frozen environment; do not infer authorization from this document.

## Hidden validation and binary pin

Completed without GPU access (`CUDA_VISIBLE_DEVICES=""`,
`NVIDIA_VISIBLE_DEVICES=none`):

- CPU `--test mixed_decoded_arena_handoff`: 2 passed.
- CPU `--lib mixed_decoded`: 29 passed. Unrelated existing CPU-only warning:
  `numeric_shard.rs::payload_liveness` is unused without CUDA.
- CPU integration-test clippy with `-D warnings`: passed.
- CUDA model `--lib --tests` clippy with `-D warnings`: passed.
- CUDA model unit-test binary `--no-run`: compiled and linked.
- Both `decoded_fixture_*` metadata-only selectors: passed, GPU hidden.
- Changed Rust files: rustfmt check passed.
- No GPU small gate or full-model execution was performed for this handoff.

Build: CUDA 12.4.99, `sm_86`, `FERRULE_CUDA_TEST_ORACLE=0`, two Cargo jobs,
`cargo test -p ferrule-model --features cuda --lib` (test profile).
NVCC: `/home/admin/wxw403883/anaconda3/envs/tao/bin/nvcc`.

Unit binary:
`/tmp/ferrule-adapter-extraction-readiness/debug/build/ferrule-model/34df30149247905c/out/ferrule_model-34df30149247905c`

SHA-256: `84fc5ef18a1dc8afe8dc271ef9dbefdc882a194f99148bd7b8074c3e7ca2573f`.
This is not a production/full-gate executable pin.

Relevant source pins at hidden build handoff (paths relative to repository):

| Path | SHA-256 |
| --- | --- |
| `crates/ferrule-model/src/transformer/standard/cuda/hybrid_rank/mixed.rs` | `54ea3983d776675f90179f9b2d85528f70063b49fd66c9a4af5594f96530d09a` |
| `crates/ferrule-model/tests/unit/transformer/mixed_decoded_gpu.rs` | `27ec068f2b582d177288435cd6d3719d01d24497ce5712631f658bb4c3f18553` |
| `crates/ferrule-model/tests/mixed_decoded_arena_handoff.rs` | `94a905c0f89daa244662ab362d1ad9aeaba58c3b67429862e8984b876c6c84b3` |
| `crates/ferrule-model/src/transformer/standard/cuda.rs` | `80eff188c2f1680c25c7fd310c4003df72735017da65008ab29e21b24e4caef4` |
| `crates/ferrule-model/src/transformer/standard/mixed_decoded.rs` | `cb4e403845a0e0621c73912a0ed72de51686b421265b93d2b94f370edcddf8fb` |
| `crates/ferrule-model/src/transformer/hybrid_parallel/mixed_decoded_arena.rs` | `9a082ee34b24525beb3206710e488400e36e3b1209dabfe022ba9480245f8d9a` |
| `crates/ferrule-backend/src/cuda/operator_owner.rs` | `8ae162a5fdf7c53665b482bfe0d1932e4cb00abd1d153aa3c8b0984bb9aae5ee` |
| `crates/ferrule-backend/src/cuda/runtime.rs` | `f7f36d50be4a5ddbd37d880da46c982d11cc7b83ed82f94e56655851d82bebfb` |
| `crates/ferrule-backend/src/cuda/allocator.rs` | `49ba582f3ad4f267ff8bbd0f6b3fe0351c1cf1a3e560e1c7d8e78c218cd6b98c` |
| `crates/ferrule-backend/src/cuda/operators/bf16_arena.rs` | `c9da2400670074b1f358250cef14603dab094854cac7d794a958b179de28b45b` |
| `Cargo.lock` | `c277518bfbdc9e8e4fbb43d0b73a2123b81682b0e7c8a04466cc8040bbdfcf78` |
| `rust-toolchain.toml` | `36d09499660ff10eeabfd80816a64414fd91934325d539105b79351f33a1381a` |

## Full-gate repinning requirements

Do not reuse the previous zero-initializing model binary's correctness or timing
claim. Rebuild/re-pin the actual full-gate executable and its complete source,
CUDA runtime/cuBLAS/CUTLASS, feature/architecture and environment closure. The
source pins above identify this local change; they are not a full artifact manifest.

Freeze the exact same checkpoint/source identities, rank/lane topology, descriptor
inventory, per-class quotas, physical KV pages, rows/batch/sequence geometry,
owner margins, oracle inputs and numeric policy as the admitted full configuration.
Keep the 3 GiB **full-grant arena** within the 22 GiB total owner admission; do not
recompute a differently shaped configuration as evidence for that admission.

Record `arena_backing_grants`, `capacity_grants`, `physical_current_bytes`,
`physical_peak_bytes`, `driver_allocation_growth_bytes`, `pinned_grants`,
`free_slot_grants`, `pooled_grants`, `unused_tail_bytes`, `slot_padding_bytes`,
`pending_lease_count`, `borrowed_key_count` and `unknown_reserved_bytes`. Dedicated
successful root growth equals its full grant; no slab reserve may be substituted
for tail or counted a second time. Preserve GDN QKV concatenation, routed 2N
pair order, separate shared projections, host retirement and existing proof gates.
