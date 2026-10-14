# Actual cached BF16 GDN replay

This is an operator regression, not a partial model factory or full-model
acceptance test. No projection, GEMM, expert, or model forward is executed.
Production conv/GDN receives the retained Hello decode.0 layer-0 QKV/A/B,
BF16 conv history, and F32 SSM, plus payload-hashed original checkpoint
conv/A_log/dt_bias parameters. F32 A_log is cast once to the declared resident
BF16 dtype, exactly as the pinned producer does.

## Independent evidence and root cause

Read-only input capture:
`target/validation/bf16-layer01-supported-retry-20261005`.
HF payload SHA256:
`6752faa07d618e884c6f4be51694aa071866fa88f4e1d0b7860e3423c9b21b07`.
Pinned Transformers 5.5.3 `modeling_qwen3_5_moe.py` SHA256:
`0f8ca77e875c31d448137c9ac291bed505c6b9927a85b5b8460af29d1e1ee647`.
Torch 2.13.0+cu129 `Reduce.cuh` SHA256:
`ef9bea8fdfd03cec2d7e7421cbe75d98fddfdaafddb4f25a1ed655c5861dfb65`.

`actual_replay.py hf` observes locals from the unmodified pinned HF recurrent
function with `sys.settrace`; both output and F32 state must equal the retained
independent capture bit-for-bit. It also observes the native BF16 L2 results.
It verifies actual projected operands and state against the independent capture,
and computes gates with the pinned HF producer expression, checking g/beta
against the independent captured g/beta.

`instrument` generates a test-only copy of the production delta kernel with
intermediate stores. The standalone CUDA12.4 replay runs both the original
entry points and this instrumented copy, requiring bit-identical output/state
and intact allocation end canaries. The generated kernel is never built into
the production provider. Only the fixed real shape is admitted by this runner.

Baseline replay exactly reproduced the old production output AND all SSM words.
Conv/history, normalized Q/K, Q scale, g, beta, decay and all decayed SSM words
matched HF. The first difference was memory reduction: 3,321/4,096 F32 values;
then correction (375), updated state (28,594), and final dot (3,330). Output
index 2603 crossed the BF16 midpoint: old `0x3912`, HF `0x3911`.

Changing only the two reduction orders reproduced every HF stage, including
524,288 F32 updated-state values and 4,096 BF16 outputs exactly. This is actual
input evidence, unlike the earlier neutral-update frozen-column observation.

## Production scope

For `rows == 1`, `64 <= key_dim <= 512`, `value_dim % 4 == 0`, and
`value_heads * value_dim >= 128`, the materialized HF F32 product tensor has
vectorized non-reduced columns and four y lanes / four accumulators per lane.
The production column owner now emulates that order locally, with separately
rounded multiply/add (no FMA contraction). This is a shape/reduction policy,
not a token/head/index correction or a replacement output.

The ABI has no cached/prefill flag: eligible **one-row prefill also uses this
order**. Captured Hello one-row prefill is bit-exact to both old production and
HF; that does not prove every one-row prefill is invariant. Multi-row scans and
shapes outside this regime retain the prior arithmetic. No chunk algorithm,
public precision profile, F32 default, BF16 boundary, SSM dtype, scratch buffer,
allocation, stream synchronization, owner/alias check, or Unknown-completion
policy changes. There is no claim of universal HF reduction/transcendental
bit parity outside the tested cases or of full40 parity/performance.

## Reproduction

Reserve GPU0 and use the repository's authenticated `run_process` supervisor
from `scripts/test_vllm_qwen35.py` for **each GPU command**. Source
`target/validation/qwen35-35b-current/toolchain/env-cuda124.sh` for native builds.
Use fresh ignored output directories; do not overwrite the independent capture.
The scripts do not themselves establish process custody or GPU quiescence.

HF export (separate process, clear LD_LIBRARY_PATH to preserve Torch CUDA12.9):

```sh
LD_LIBRARY_PATH='' CUBLAS_WORKSPACE_CONFIG=:4096:8 /opt/conda310/bin/python -B \
  crates/ferrule-backend/tests/recurrent_bf16/actual_replay.py hf OUTPUT
```

`actual.sha256` pins every input and expected file for the Rust regression.
Those expected values come only from HF, never the corrected production kernel.

```sh
FERRULE_GDN_ACTUAL_FIXTURE=OUTPUT cargo test --locked -p ferrule-backend \
  --features cuda --test cuda_recurrent_bf16 \
  gpu::actual_cached_decode_matches_pinned_hf_state_and_output_exactly \
  -- --exact --ignored --nocapture
```

For internal-stage diagnosis, run `actual_replay.py instrument OUTPUT`, then
compile `actual_replay.cu` with `-std=c++17 -O3 -arch=sm_86 --cudadevrt=none
--cudart=shared`, include paths `crates/ferrule-backend/native/cuda` and `OUTPUT`,
and the validated CUDA12.4 linker directory. Run its binary as `OUTPUT LABEL`.
Compare without GPU work using:

```sh
CUDA_VISIBLE_DEVICES='' /opt/conda310/bin/python -B \
  crates/ferrule-backend/tests/recurrent_bf16/actual_replay.py \
  compare OUTPUT --label LABEL --require-exact
```

Additional dedicated standalone regressions:

- `reduction_matrix.py OUTPUT`: exports 55 native Torch CUDA F32 reductions;
  compile/run `reduction_matrix.cu OUTPUT` against production `column_sum4`.
  Includes tails, key dimensions 64–512 and several head/value layouts.
- `stability.cu`: compares the current kernel with an archived pre-fix kernel
  in the same TU. Supply `prior.cuh` made from the archived header (SHA256
  `da079c6199d884173798aca67effd462c9115414bc46559dc8891d5dbbca85b2`), renaming
  `hf_bf16` to `prior_hf_bf16` and `ferrule_core_bf16_` to `prior_core_bf16_`.
  It checks multi-row prefill/continuation and excluded single-row shapes.

Fresh receipts, comparisons and final release evidence for this investigation
are in `target/validation/gdn-actual-20261006`. Original references and all
existing tolerances remain unchanged.

## Matched old/new performance (measured 2026-10-06)

`performance.py prepare` is CPU-only, forces GPUs hidden for all subprocesses,
verifies the archived old kernel and captured payload hashes, and emits
`prior.cuh` **only in the chosen ignored output directory**. It compiles old/new
in the same CUDA12.4 `-O3 -arch=sm_86` TU, saving full PTXAS registers, stack and
spill reports. It also builds the expanded stability regression (107 cases,
including 10 otherwise-eligible one-row cases with `Hv*Dv < 128`). Host-only
`--describe` / `--list-cases` paths validate coverage without touching a GPU.

Prepared local artifacts:
`target/validation/gdn-performance-fenced-20261006/prepared.json`.
The earlier unfenced preparation is stale and must not be reused.
CPU preparation observed old delta **40 registers**, new delta **45 registers**;
both have **0 stack bytes / 0 spill stores / 0 spill loads** in this build.
These resource counts are not event timing or a performance acceptance result.

The benchmark uses Dk=128, Dv=128, Hk=16, Hv=32 with rows 1/5/32:

- Row 1: actual Hello cached decode state and convolved inputs. Old output/state
  must match the historical producer; new output/state must match pinned HF.
- Rows 5: actual capital prefill convolved QKV/A/B, with zero initial SSM.
- Rows 32: the same five captured rows cyclically tiled to 32, explicitly a
  controlled microbenchmark rather than a genuine 32-token HF/model trace.
  Old/new multi-row output and SSM must match bit-for-bit.

Old/new share the **same input, state and output addresses**. Every invocation
restores identical initial SSM using a same-stream D2D copy **before** its start
event. State never drifts towards a trivial steady state. Each CUDA event pair
brackets one production GDN launch (including stream scheduling gaps), not
copies, allocation, validation, conv, or projection. No CUDA graph substitution.
This is a warm/reset-touched cache protocol, not cold-cache or serving timing.

There are 20 warmups per variant, then 30 paired samples in alternating AB/BA
order, each containing eight individually event-timed invocations. Raw timings,
paired ratios, medians/p95s and runtime kernel attributes are emitted. Full state
and output are checked before timing and after **every sample**, outside events;
input immutability and allocation guards are checked outside timing. All current
source/binary/operand hashes are rechecked before execution; stale preparation
fails closed. The run records clock/power/temperature snapshots without changing
clocks, uses the existing authenticated supervisor, and checks drain/quiescence.

CPU preparation/check commands (use an empty output directory for preparation):

```sh
CUDA_VISIBLE_DEVICES='' /opt/conda310/bin/python -B \
  crates/ferrule-backend/tests/recurrent_bf16/performance.py prepare \
  target/validation/gdn-performance-fenced-20261006
CUDA_VISIBLE_DEVICES='' /opt/conda310/bin/python -B \
  crates/ferrule-backend/tests/recurrent_bf16/test_performance.py \
  --prepared target/validation/gdn-performance-fenced-20261006
```

**GPU owner only**, after all GPUs are idle/released (not during CPU preparation):

```sh
. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
/opt/conda310/bin/python -B \
  crates/ferrule-backend/tests/recurrent_bf16/performance.py run \
  target/validation/gdn-performance-fenced-20261006 \
  --evidence target/validation/gdn-performance-fenced-rerun-20261006 --execute-cuda
```

The evidence directory must not exist. The authenticated supervisor bounds the
stability process to **60 seconds** and performance process to **120 seconds**,
plus bounded cleanup and post-run checks. It will save `timing.json`, PTXAS log
pins, supervisor receipts and `release.json`. Preparation does not run these GPU
tests. The successful recorded run is
`target/validation/gdn-performance-fenced-run-20261006`.

| Rows | Old median (us) | Current median (us) | Change | Old p95 (us) | Current p95 (us) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 51.968 | 66.944 | +28.8% | 52.224 | 67.200 |
| 5 | 222.976 | 230.208 | +3.2% | 227.072 | 232.832 |
| 32 | 1248.512 | 1351.872 | +8.3% | 1277.824 | 1364.736 |

These are **measured operator slowdowns**, not throughput projections. Multi-row
arithmetic is unchanged, but the compiled kernel and resource usage changed;
the timing alone does not isolate why each multi-row measurement slowed down.
107 stability cases passed (2,914,300 state/output comparisons), including all
10 below-threshold cases. Full output/state, input immutability and guards passed
in the benchmark; old/current use 40/45 registers, zero stack/spill stores/spill
loads, runtime local/shared bytes zero, binary version 86.

The original run failed its old-output exact guard **before timing**. A diagnostic
retry localized the corruption to the final 256 outputs. Pageable seed H2D can
return after host staging, before device DMA completes; the nonblocking benchmark
stream did not wait for that default-stream transfer. `performance.cu` now fences
the default stream once after seed upload, entirely **outside timing**. All exact
guards remain unchanged. Both failed receipts are retained in
`gdn-performance-run-20261006` and `gdn-performance-diagnostic-run-20261006`;
the successful evidence uses a freshly pinned preparation. Authenticated drain
and three all-eight-GPU idle samples passed for each run.

### Load-reuse assessment

The selected single-row path currently decays state, reads it again for memory,
updates state, then reads it again for the final dot. It also recomputes normalized
K/Q in the reduction helper. A potential refactor could accumulate newly decayed
or updated products into the same 4-by-4 accumulator slots while preserving each
slot's key order and the exact combine tree. However, keeping 16 accumulators
live (or caching 128 state/Q/K values) may increase registers, spills or reduce
occupancy. A single scalar running sum would change the proven arithmetic and
is not a valid optimization. No such change is made without matched event
measurements and repeat exact intermediate/state/output checks.


## Actual capital multi-row chunk investigation

`chunk_observation.py` replays only the frozen five-row layer-0 operators. It
verifies every frozen payload and production-stage hash from
`target/validation/capital-layer0-current-20261006/GDN-HANDOFF.json`, runs the
pinned CUDA12.4/sm86 native replay into fresh evidence, and observes the original
Torch 2.13.0+cu129 / Transformers 5.5.3 chunk and recurrent functions in a separate
authenticated child. No model/projections/full40 are executed. Original retained
HF outputs and F32 state are read-only exact guards, not regenerated references.

The final evidence is `target/validation/gdn-chunk-stages-final-20261006`:

- Native conv/history/output/state and **every native traced stage** match the
  retained producer bit-for-bit; allocation canaries pass.
- Fresh HF conv/Q/K/V/g/beta and chunk output/state match the independent capture
  bit-for-bit. Post-L2 Q/K, scaled Q and all actual scan decays are now independently
  proven exact to the appropriate HF stages.
- The first directly comparable serial difference is token0 final F32 reduction:
  3,333/4,096 unequal. Token0 decayed state, memory and correction are exact.
  Token1 decayed state is still exact; memory then differs in 3,388/4,096 entries.
- The chunk algorithm also reassociates decay **before its first matrix product**:
  token1 `exp(prefix_g[1]-prefix_g[0])` differs from actual `exp(g[1])` in 4/32 heads.
  For head3 these are `0x3e076bc8` vs `0x3e076bc6`. This is evidence of a different
  arithmetic sequence, not a bad gate or an isolated exponential implementation.
- With zero initial state, transformed chunk value is mathematically equivalent
  to the serial correction, not bitwise equivalent: differences by token are
  0, 1,363, 1,367, 1,637, 1,894 of 4,096. Output-dot differences start at token0.
- At token2/head17/column108 (flat10476), F32 dots are production `0x33f17ffe`, HF
  recurrent `0x33f18002`, HF chunk `0x33f18001`. BF16 outputs are `0x33f1`, `0x33f2`,
  `0x33f2`. Production differs from chunk in one of 20,480 BF16 output words and
  302,175/524,288 F32 state words (max 9.5367431640625e-7).
- HF recurrent happens to match all chunk BF16 outputs on this fixture, but
  **297,454 F32 state words still differ**. Removing the `rows == 1` guard would
  not implement the chunk algorithm and is not proposed as the prefill fix.

`hf-stages.safetensors` contains the prefix g, decay matrix, triangular matrix
before/after transform, transformed value, cumulative decayed keys, v_prime,
v_new, state, F32 output, and operands/results of all eight original HF matmuls.
`sys.settrace` observes statement boundaries; `TorchDispatchMode` redispatches
`aten.matmul` unchanged then clones its actual operands/results. All 53 stages
shared with the earlier no-matrix-observer run are bit-identical. This establishes
observation stability, not CUDA12.4/cuBLAS numerical parity with Torch CUDA12.9.

### Bounded reproduction

After exclusive GPU ownership and all-eight-GPU idle confirmation, use a fresh
empty evidence directory (the runner enforces this and authenticates native
libraries, supervisor drain, and three final quiescence samples):

```sh
. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
/opt/conda310/bin/python -B \
  crates/ferrule-backend/tests/recurrent_bf16/chunk_observation.py run \
  target/validation/gdn-chunk-stages-rerun-20261006 --execute-cuda
```

Native child: **60 seconds**; pinned HF child: **120 seconds**; bounded supervisor
cleanup and three one-second-spaced final GPU samples are additional. CPU checks:

```sh
CUDA_VISIBLE_DEVICES='' /opt/conda310/bin/python -B \
  crates/ferrule-backend/tests/recurrent_bf16/test_chunk_observation.py \
  --evidence target/validation/gdn-chunk-stages-final-20261006 \
  --without-matrix-observer target/validation/gdn-chunk-actual-retry-20261006
```

### Implementation boundary — not implemented

A legitimate chunk path is algebraically possible with bounded workspace. It is
**not** a safe kernel-only change to the current scratch-free chronological ABI.
A conservative, unoptimized single-chunk design (C=64, H=value_heads) can reserve:

- Six F32 `[H,C,Dk]` arrays: scaled Q, K, K_beta, K_beta*exp(prefix), transformed
  decayed K, and Q*exp(prefix).
- Seven F32 `[H,C,Dv]` arrays: V, V_beta, transformed V, state projection, V_new,
  inter-chunk output, and intra-chunk output.
- Four F32 `[H,C,C]` arrays: decay, triangular transform, attention, and temporary
  triangular products. A separate launch/reduction preserves each transform row's
  dependency and avoids cross-block in-place races.
- Three F32 `[H,C]` arrays for g/prefix/beta and one F32 `[H,Dk,Dv]` state-product
  temporary. Existing S remains F32 and is updated only after its readers finish.

Logical extra bytes are
`4*H*(6*C*Dk + 7*C*Dv + 4*C*C + 3*C + Dk*Dv)` = **17,850,368 bytes**
(17.0234375 MiB) at H32/Dk128/Dv128. This is a conservative proposed layout, not a
measured or validated minimum; allocator alignment and any matmul-provider scratch
are additional. Process chunks sequentially to bound memory independently of row
count. HF precomputes across all chunks in a batch: changing that batch dimension
may change library matmul dispatch, so multi-chunk bit parity still needs proof.

Required integration boundary:

1. Checked scratch layout/byte budget and explicit owner-managed F32 workspace
   operand in the recurrent plan/ABI; overflow, extent, owner and writable-alias
   checks must cover workspace as well as state/output. Do not allocate secretly
   inside an operator or reuse BF16 output/F32 state as hidden scratch.
2. Preserve BF16 L2 and beta boundaries, promote before prefix/triangular/matmul
   math, keep F32 state and separate HF multiply/add boundaries. Do not replace
   those with FMA or a GEMM beta epilogue without stage-level evidence.
3. Implement/validate the eight F32 matmul stages and triangular reductions. The
   inspected public `linear_f32_into` contract is artifact-weight TF32x3 CUTLASS,
   explicitly **not bitwise IEEE SGEMM**; it is not a proven drop-in for these
   dynamic batched operands. Existing buffer/stream ownership can be reused;
   no suitable pinned-HF-parity batched primitive was established in this scope.
4. Model-side allocation/lifetime integration is necessary: `bf16_hybrid.rs`
   currently allocates only convolved/core/output BF16 buffers and supplies no
   workspace to GDN. Recurrent `submit_operator(1,...)` must account for the real
   multi-launch sequence; a failure after partial submission must retain or
   quarantine workspace **and state** through existing Unknown retirement/fences.
5. Dedicated stage tests must cover nonzero F32 initial state, rows2/5/32/63/64/65,
   multiple chunks, tail padding, ownership/alias/budget rejection and partial
   completion. The current real fixture is zero-state rows5, not proof for those
   cases. Measure both scratch peak and event performance before integration.

This crosses the permitted recurrent/test-only write scope into workspace/model
integration. Production multi-row arithmetic, public profiles, reference/state
policy and tolerances therefore remain unchanged. No register/load-reuse
optimization was implemented: the measured slowdown justifies a future matched
experiment, not an unmeasured rewrite of the proven reduction tree.


## Implemented backend chunk slice

The earlier implementation-boundary section above records the pre-implementation
assessment. The opt-in backend slice is now implemented and actual/synthetic exact
gates pass; model/default dispatch remains unchanged. See [CHUNK.md](CHUNK.md)
for the typed workspace API, resolved dense reduction-order defect, bounded
reproduction commands, resource/lifetime contracts and integrator handoff.
