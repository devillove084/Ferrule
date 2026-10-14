# Mixed decoded cache — model handoff after P1/P2 review

## Status and boundary

Model candidate remains **internal opt-in / public default off**. Pair API is
now connected. No backend/runtime/composition/paper edits in this model task;
no GPU numeric execution, full40 rerun or performance acceptance.

## Pair and fixed projection data paths

`standard/cuda/hybrid_rank/mixed.rs` uses:

- `decode_numeric_bf16_pair_into(op, [gate, up], target_2nk)` on routed pair
  misses. The two original GPU decode launches fill one retained `[gate; up]`
  allocation. No temporary half-buffers, model subviews, DMA or host decode.
- `execute_bf16_pair_view(..., Bf16GemmLayout::contiguous(M, 2*N, K))` on the
  same prepared full-2N view, followed by the original backend split.
- Fixed attention/GDN/shared/dense projections are decoded at preparation and
  pinned. GDN QKV keeps its existing combined artifact; shared/dense gate/up
  remain independent GEMMs. Single projections use `execute_bf16_view`.
- Only a pre-submission physical-admission miss returns to the original native
  decode/GemmEx path. Never retry after decode/GEMM error, Unknown or panic.

## P1: actual backing custody, not logical eviction credit

`standard/mixed_decoded.rs` retains every allocated backing for the image:

1. `Cache<K>` remains the only LRU. Its extracted `evict_one_completed` is also
   called by existing `make_room`; no second victim ordering/search algorithm.
2. LRU removal moves an already-completed root into a pool keyed by **exact BF16
   element count and grant**, without Drop, allocator retirement or sync.
3. A miss reuses an exact matching root, overwrites it with the original GPU
   decode, and retains the new lease to the existing outermost healthy
   synchronous completion. Repeated borrows share that lease.
4. If physically full, request one victim from the existing completed-only LRU.
   If its backing still does not fit, use native execution. No scan/retry loop,
   no extra event, host synchronization, buffer subview or allocator API needed.
5. Incomplete/borrowed backing cannot be pooled/reused. Healthy failed decode
   moves its root to the pool; it does NOT refund physical backing grants.
   Failed allocation tickets also retain grant credit for the whole image:
   backend failure might already have queued allocator retirement.
6. Unknown/panic quarantines active AND pooled roots, pending grants and segment
   reserve. No retry/reuse/refund even after a late fence. Quarantined custody
   retains roots on Drop; normal whole-owner teardown uses the existing fences.

The physical charge is ADDITIVE:

```
all backing grants (active + pooled + failed/pending + unallocated fixed pins)
+ cache-induced segment slack
+ outstanding worst-case segment-slack reservation
```

Before each new allocation, reserve `max(grant, 64 MiB) - grant` slack. The
current backend allocator creates at most one 64 MiB standard segment or one
grant-sized huge segment per ordinary request; a source-contract CPU test
forces review if those rules change. After allocation, settle slack to
`max(owner_allocator_driver_allocation_delta - grant, 0)`. Count grants even
when allocated from pre-existing holes. Never use `max(grants, driver_growth)`:
that would omit the sum of old-hole backing and new-segment slack. Never credit
unrelated frees or retirement. No CUDA driver query is added on hits.

This conservative reserve is charged INSIDE the requested 3 GiB candidate cap,
not hidden outside the report or deducted from provider workspace. Native
owner allocator/context margins remain untouched. Conservatism can reduce
routed cache coverage; incompatible/full misses explicitly use native, and
failure to populate fixed pins rejects preparation. No full-model expansion.
These are admission bounds, not a measured whole-process memory peak proof.

Rank-performance adds active/pooled grants, driver-segment growth, charged
segment slack, physical charged/peak, reuses and native fallbacks to existing
hit/miss/decoded-byte/eviction observations. Profiling stays opt-in; values are
owned accounting, never OS-free memory. Provider workspace is unchanged.

## P2: complete admission identity in the claim

`decoder/hybrid_cuda.rs::HybridCudaPreparationContext::eq` now compares
`program.decoded_cache` in addition to existing fields. The admission derives
exact Eq over:

- enabled/disabled;
- requested cache capacity, fixed pins, routed working set and entry count;
- physical pages;
- total device capacity;
- owner margins.

Tests hold estimate/projection grants identical and independently mutate every
field above; old/different claims are rejected. Equal grants are not authority.

## API / reseal contract

The internal API signature is unchanged:

```rust
let candidate = native_program.with_decoded_weight_cache(
    &resources, physical_pages, requested_cache_grants,
    device_capacity, owner_margins,
)?;
let budget = candidate.decoded_weight_cache_budget().unwrap();
let estimate = candidate.operation_estimate(&resources, physical_pages)?;
```

`operation_grants` includes the whole cache ceiling once. Provider storage and
external reserves stay unchanged. `cuda_claim_inputs` includes it in workspace;
claimless and claimed preparation both validate the enlarged device budget.
Runtime/composition must **recreate/reseal the report and claim from the same
candidate program and full admission config**. Reusing a native claim or a
same-grants claim with different capacities/margins now fails closed.

## Source-derived TP2 ledger

For context 128, max rows 8, sessions 2, page size 16, state copies 5, **64 physical
slots** (32 logical pages), both rank metadata calculations reported:

| Category | Bytes/rank |
| --- | ---: |
| Native model grants excluding owner margin | 19,220,262,912 |
| Pinned BF16 grants | 1,405,091,840 (1340 MiB) |
| Largest full routed-layer grants | 805,306,368 (768 MiB) |
| Requested cache ceiling, now INCLUDING segment slack | 3,221,225,472 |
| Original owner margin | 134,217,728 |
| Total admitted | 22,575,706,112 |
| 22 GiB limit | 23,622,320,128 |
| Remaining | 1,046,614,016 |
| Remaining with original 8 MiB diagnostic reserve | 1,038,225,408 |

20,480 routed descriptors are metadata, not allocations. 1732 MiB is the
remainder after pin GRANTS, not guaranteed usable routed backing: segment
slack/failed tickets also consume it. The final metadata test passed for both ranks in 83.88s, including the
negative-page check. Earlier pre-fix attempts timed out; negative-page
rejection now occurs before expensive estimation.

## GPU-hidden validation and exact selectors

CUDA 12.4.99, SM86, pinned environment from
`target/validation/mixed-routed-down-m1-small-20261010-01/environment.json`;
`CUDA_VISIBLE_DEVICES=''`, `NVIDIA_VISIBLE_DEVICES=none` throughout.

Observed:

- 15 metadata/admission/physical-cache CPU tests passed in CUDA-linked binary.
- Full-admission claim identity CPU test passed.
- Pair/shared descriptor metadata + host-payload retirement test passed.
- Real TP2 header-only ledger test passed for both ranks (83.88s, GPU hidden).
- Existing expert-cache tests: 8 passed; timing: 8 passed / 1 calibration ignored;
  upload-proof tests: 6 passed.
- `cargo check -p ferrule-model --features cuda --tests` passed.
- `cargo test -p ferrule-model --features cuda --lib --no-run` compiled/linked.
- CUDA model `cargo clippy --lib --tests -- -D warnings` passed.
- Initial link attempt timed out at 240 seconds. An unpinned command then
  failed to find nvcc. Correct pinned retry completed in 2m06s; no source
  workaround or GPU visibility change was used.

GPU owner must build/reseal its final source/binary after all agents finish.
Current compiled artifact (not a release/quality promotion):
`target/catalog-tp2-cuda/debug/build/ferrule-model/452649c98ed34710/out/ferrule_model-452649c98ed34710`

Set `FERRULE_RUN_MIXED_RANK_GPU=1`, `CUBLAS_WORKSPACE_CONFIG=:0:0` and the sole
GPU owner's two assigned visible devices. Each selector below is run separately
with `--exact --ignored --test-threads=1`; no GPU run was performed here:

```
transformer::standard::cuda::hybrid_rank::mixed::decoded_gpu_tests::decoded_pair_single_cold_hot_reload_exact_and_no_cache_dma
transformer::standard::cuda::hybrid_rank::mixed::decoded_gpu_tests::decoded_fixed_pins_prepare_once_and_shared_stays_separate
transformer::standard::cuda::hybrid_rank::mixed::decoded_gpu_tests::decoded_pair_unknown_and_panic_retain_physical_backing
transformer::standard::cuda::hybrid_rank::mixed::decoded_gpu_tests::decoded_physical_full_native_fallback_before_submit_exact
```

The first covers pair/down M1/M5 exact output vs unchanged native, cold/hot
launch delta, no cache DMA/artifact upload, source-file removal, and forced
completed backing reuse without new cache grants/driver growth. The second
covers prepare-once shared pins and preservation of separate GEMMs. The third
injects typed Unknown and panic immediately after successful pair decode and
checks no native retry, no late-fence recovery and no backing release on Drop.
It intentionally quarantines allocations until test process exit. The fourth
forces a physically full cache before routed work and checks exact unchanged
native pair/down execution at M1/M5 with no new cache grants or segment growth.

Host-only selectors (GPU hidden):

```
transformer::standard::cuda::hybrid_rank::mixed::preparation_tests::decoded_claim_tests::decoded_claim_compares_full_admission_even_when_grants_match
transformer::standard::cuda::hybrid_rank::mixed::decoded_gpu_tests::decoded_fixture_descriptors_preserve_pair_and_separate_fixed_roles
```

These small synthetic gates do NOT replace independent real-HF words, combined
GDN QKV/full program exactness, full40 + six recipes, physical peak observation,
or performance acceptance. Default off until all unchanged gates pass.
