# Prepared routed slots and batch gather — model source freeze

2026-10-12. This is a model/test source freeze, not a binary, backend, GPU
numerical, full40 or throughput acceptance. The supplied fused 6.51 tps remains
an input baseline; no new E2E result is inferred from fewer scans or launches.
Verify `mixed_route_batch.sha256` from the repository root before validation.

## Lookup scope and source identity

The pre-change `mixed.rs` SHA-256 was
`c0cd7c6d4ac75e082801b2e9d76bc43172af7a131879c758ee9f153835aa25f8`.
Its three `.position()` calls per selected expert scanned the **entire image's
projection Vec**, not a layer-local projection directory. Their early-exit
cost depends on the original Vec order. No E2E speedup follows from this alone.

Preparation now builds sorted layer headers and compact expert-indexed
`[gate_slot, up_slot, down_slot]` arrays. Generation, complete original bindings,
paired numeric source snapshots, roles/residency, shard shapes, duplicate slots,
duplicate roles, holes and bounds are validated before uploads. No projection
Vec is reordered, so decoded descriptor/cache-key slot identity is unchanged.
Runtime uses the frozen generation and checked slots, not current file metadata
or a newly reconstructed shape-only binding. Post-preparation FP8 retirement
and removal of the checkpoint file remain supported; no new runtime source I/O.

## Integrated backend contract and compatibility

The actual `standard_bf16` API used, not an invented interface, is
`CudaBf16RouteMetadata`, `allocate_bf16_route_metadata`,
`prepare_bf16_route_metadata`, `gather_bf16_routes_into`, and
`weighted_combine_bf16_routes_into`. Allocation precedes caller custody;
preparation is one same-compute-stream H2D of `[i32[R] | f32bits[R]]`.

- One expert-ascending plan, stable old row order within every expert bucket.
- One exact `[R,hidden]` BF16 gather root and one existing kind2 gather kernel.
- Each expert consumes `.slice(begin*hidden, count*hidden)`. Original SameMe2N
  pair/fused-tail/down GEMM M/N/K and packed-output offsets are unchanged.
- The original BF16 combine reads the **same** opaque metadata allocation;
  original F32 weight bits, BF16 product/add rounds and expert fold order remain.
- Input, gather and packed roots are held before use. Immutable metadata source
  and device buffer enter image custody before preparation. Existing outer-depth
  completion also runs for routed images with decoded caching off. Metadata is
  cleared only after the original healthy outer fence; Unknown/panic retains
  the image, metadata and BF16 roots, with no native retry or GEMM replay.
- Native/legacy diagnostic observers still select the old split pair tail.
  Paired gate/up `mixed.linear.input` events borrow complete expert input views
  at the real batch offsets. Observer state does not select a different gather.
- The delivered opaque backend API currently limits M<=8/top-k<=8/R<=64.
  Larger originally valid M/top-k retains old gather/combine **in the same
  expert loop**. No new public profile, factory, collective or layer loop.

## Current ledger, not payload-only savings

Gather payload remains exactly `2*R*hidden` bytes. Its **single allocator grant**
must be rounded once; allocator size classes are not subadditive. For buckets
`[5,1,3,2]`, hidden=4096: old gather grants sum to 122,880 B; the new grant is
131,072 B, an 8,192 B increase despite equal payload.

The neutral ledger retains the old full activation/tail/provider/cache envelope
and separately reserves `grant(8*R)` for opaque metadata (512 B at R=64).
Host staging includes overlapping model IDs/weights and opaque immutable H2D
source (`16*R` bytes), compact groups (`3*sizeof(usize)*G`, 768 B at G=32),
per-image table/custody Vec headers, per-layer headers, and original three-slot
expert arrays (6,144 B per 256 experts on 64-bit hosts). The existing 22 GiB
capacity admission is not bypassed. No released-tail or shortened lease
lifetime is credited. Rank-performance-off wiring and per-key observations are
unchanged; no extra per-key snapshot/report allocation was added.

## Gates and missing evidence

CPU-only tests cover independent old-vs-batch ordering, M1/M5, R=64, overlapping
routes, decode, ascending experts, exact F32 bits including -0/subnormals, bucket
rows, complete BF16 view slices, SameMe2N/down geometry, slot failures,
compatibility bounds, real allocator grants and synthetic capacity rejection.
`mixed_route_batch_frozen.rs` contains literal independent old gather/route
fixtures, including raw signed-zero, infinity and NaN words; no BF16/F32
conversion is used to compare the gathered fixture.

GPU gates are defined in `mixed_route_batch_gpu.rs`, with a separately frozen
pre-batch routing reference in `mixed_route_batch_legacy.rs` that never calls the
new planner/opaque API. They compare **raw u16 local partials**, cold/hot cache,
M1/M5 overlap/decode, source retirement, complete observer input offsets, one
8*R metadata H2D, and Unknown immediately after the actual whole-route gather
(cache enabled and disabled). Partial arithmetic uses unchanged original GEMMs;
the independent CPU fixture is a gather oracle, **not** a GPU partial golden.
A real model R=64 partial fixture is still required for GPU acceptance; the
existing synthetic model fixture has two experts and a maximum M of five.

Validation here: 43 CPU tests passed, five real-header/frozen/manual gates
ignored; targeted CPU Clippy with `-D warnings` and changed-file rustfmt checks
passed. CUDA-feature type/build validation is blocked by missing `nvcc`.
GPU tests have not been type-checked or executed; no GPU equality/counter,
real-header 22 GiB acceptance, full40 or E2E performance claim is made.
`FERRULE_MIXED_HEADER_DIR` was not provided. Metadata-only source/performance
snapshots and GPU partial goldens are **null/unmeasured**, never synthetic zero
or a successful measurement. GPUs remained hidden throughout validation.
