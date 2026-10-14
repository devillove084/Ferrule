# Ordinary BF16 offset RMSNorm: exact first-divergence fix

## Actual evidence and cause

Frozen source: `target/validation/bf16-resident-control-20261007/frozen-first-norm`.
Hello hidden0..15 and layer16 attention/residual are exact; the first same-input
norm mismatch is post-attention layer16, output452. Layer/index are provenance,
not implementation conditions.

The checkpoint is native BF16. Reading its safetensors header locates the weight
payload at absolute byte12170912 (data offsets11863104..11867200); all4096 bytes
match the frozen weight. The existing BF16 model path expands input and weight,
executes F32 offset RMSNorm, then casts BF16. Unmodified pinned Qwen3.5 HF dispatch
confirms F32 square -> mean -> epsilon add -> rsqrt -> input multiply, followed
by F32 weight expansion -> (1+weight) -> multiply -> sole final BF16 cast.
There is no intermediate BF16 normalized value or BF16 (1+weight).

The instrumented old kernel reproduces the uninstrumented result exactly and
retains the original one-word mismatch. Earliest differing stage:

| Stage | Existing bits | Pinned HF bits |
|---|---|---|
| mean | 0x3f88be2a | 0x3f88be29 |
| mean+epsilon | 0x3f88be32 | 0x3f88be31 |
| inverse RMS | 0x3f77ae4e | 0x3f77ae4f |

All actual squares are exactly representable in F32 and equal HF squares; all
2048 F32 affine values match. Thus this fixture isolates reduction order, not
weight loading, F32 affine semantics or a BF16 boundary defect. Old normalized/
weighted values differ1567/1358 words before final rounding leaves one output
mismatch: -0.10009765625 versus -0.1005859375.

At rows1,width2048, old reduction uses256 strided lanes; pinned HF uses512 lanes,
four contiguous values/lane and its halving tree. The shared helper also handles
row-dependent block geometry, virtual materialized-F32 alignment and tails.
The candidate and final production kernel match every actual intermediate/output.

## Production scope and contracts

- `bf16_rms_reduction.cuh` extracts only the already validated gated norm mean
  helper. Gated wrappers and its epilogue are retained; no GDN math/state change.
- `bf16_boundary.cuh` kind7 implements BF16 input/weight/output offset norm,
  explicit F32 RN operations and one final cast. The existing kinds1..6 remain.
- `CudaOperators::offset_rms_norm_bf16_into` takes typed BF16 buffers and an
  `F32RowsLayout`; exact extents, owners, finite positive epsilon and writable
  disjointness are checked. Read/read aliasing remains legal. One launch, no
  submission allocation, copies, scratch or host synchronization.
- Only `bf16_norm`'s one-plus-weight branch switches to this typed operator.
  One output allocation remains at the model boundary; conversion intermediates
  are removed. Weight loading, trace hooks, model profiles and other branches
  are unchanged. F32 norm implementation is byte-identical.
- Widths1..4096 use the validated Torch tree. Larger widths retain the old256-lane
  reduction/epilogue arithmetic directly on BF16 storage; no wider HF-parity
  promise. Tests4097/8192 enforce old/new bitwise stability.
- NaN/Inf and square overflow propagate; finite inputs alone need not yield finite
  intermediates. Nonfinite tests check classification and exact non-NaN values,
  not NaN payload identity. No clamping or acceptance-tolerance change.
- Default Qwen offset weights are zero, yielding affine1; weight1 yields affine2;
  weight-1 cancels. Small offsets, epsilon extremes and signed zeros are covered.

## Tests and measured operator latency

Evidence root: `target/validation/bf16-ordinary-norm-20261007`.

-105 ordinary cases:96 finite Torch-tree stage/output exact,3 nonfinite/overflow
 cases,6 unchanged wider fallback cases. All input/weight guards preserved.
-137 gated cases pass, including original Capital/Hello and synthetic boundaries.
-Seven selected Rust GPU tests pass:105-case typed actual/synthetic replay,
 owner/extent/epsilon rejects, real typed writable-alias slices and capture-safe
 resource counters, existing F32 norm math/guards, and existing GDN norm pipeline.
-Five CPU evidence checks pass; strict backend CUDA all-tests clippy is clean;
 model CUDA compilation succeeds. No full40/model acceptance run.

Actual1x2048 matched events:10 warmups per variant,30 alternating AB/BA pairs.
Old resident graph (two expansions + F32 norm + cast) versus new single kernel:
median13.312/6.144us; p95 14.336/7.168us. Allocation cost excluded, full result and
input checks outside timing. Old norm kernel16 registers, new27; both zero
local/stack/spill bytes; new shared2048 bytes. These are graph latencies, not a
pure reduction-kernel comparison or model throughput claim; old output retains
its one-word error. Clocks were not locked or logged during timing.

## Bounded reproduction

From the repository root, prepare fresh directories with GPUs hidden, then use
the existing authenticated supervisor only with exclusive GPU ownership:

```sh
. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=void /opt/conda310/bin/python -B crates/ferrule-backend/tests/ordinary_norm/suite.py prepare target/validation/ordinary-norm-rerun
/opt/conda310/bin/python -B target/validation/gdn-chunk-splitk-20261006/owned.py target/validation/ordinary-norm-rerun/jobs.json target/validation/ordinary-norm-rerun-receipts
```

HF stages are observed via Torch dispatch around the unmodified pinned producer,
not a replacement Python norm. Native observer output must equal the normal
production output. Exact frozen original expectations are never rewritten.
The oracle remains Torch2.13cu129/Transformers5.5.3 in a separate process; native
runtime stays CUDA12.4/sm86. Final pins/custody and release are in `FINAL.json`.
