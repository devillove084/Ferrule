# Opt-in backend BF16 chunk GDN

Implemented in `native/cuda/core/recurrent_bf16_chunk.cuh`, with separate native
ABI, Rust FFI, checked `Bf16ChunkPlan`, and the dedicated operator module
`cuda::operators::recurrent_bf16_chunk`. Existing `recurrent_bf16.cuh`, model and
composition dispatch are not modified. There is no new public precision profile,
CPU fallback, external-BLAS handle, or launch-time allocation.

## Integrator contract

```rust,ignore
use ferrule_backend::cuda_plan::Bf16ChunkPlan;
use ferrule_backend::cuda::operators::recurrent_bf16_chunk::CudaBf16ChunkWorkspace;

// Admission/allocation phase, not the operator submission hot path:
let plan = Bf16ChunkPlan::new(layout, caller_scratch_budget_bytes)?;
let mut scratch = ops.zero_f32_buffer(plan.workspace_elements())?;

// Multi-row: existing typed Bf16GatedDeltaNetBuffers, plus explicit scratch.
ops.gated_delta_net_bf16_chunked_into(
    buffers,
    Some(CudaBf16ChunkWorkspace::new(&mut scratch, plan)?),
    plan,
)?;
```

For **cached rows=1**, existing `Bf16ChunkPlan::new` still constructs a zero-budget
plan and delegates to the existing recurrent path with `None` workspace.
For **prefill, including rows=1**, use `Bf16ChunkPlan::new_prefill`: it explicitly
selects padded chunk arithmetic and requires full workspace. Execution phase,
not row count or state contents, must select the constructor. See `PREFILL_ONE.md`
for the API, exact actual M1 evidence and owner-lifetime handoff.

Multi-row capability: positive Dk/Dv <=512; positive Hk/Hv with Hv divisible by
Hk; Hv <=65535; checked packed operand/state extents and scratch bytes <=INT32_MAX.
These are admission limits, **not a promise of universal external-BLAS bit parity**.

- Packed convolved BF16 QKV, BF16 a/b/A_log/dt_bias and output; S is actual F32
  `[Hv,Dk,Dv]`. No implicit zeroing of S, dtype reinterpretation or BF16 SSM.
- Scratch is an exact-length, caller-owned `CudaF32Buffer`. Query
  `workspace_elements()` / `workspace_requirements()` before allocation.
- At C64/Hv32/Dk128/Dv128: **4,462,592 F32 elements = 17,850,368 bytes**
  (17.0234375 MiB), alignment4. This is logical scratch, not allocator peak.
  It is independent of sequence length; all chunks reuse the same panels.
- Native ABI: 104-byte `FerruleCoreBf16ChunkArgs`; `workspace_f32` at offset80,
  `workspace_elements` at88, stream at96. Private panel offsets are not a public
  caller contract. Native and Rust both validate extents/budget; native additionally
  rejects null/misaligned/overflowing/overlapping address ranges before launch.
- All operands must belong to the submitting operator owner, not merely the same
  device ordinal. Read operands may alias each other; state/output/entire supplied
  workspace must be disjoint from all other operands and each other.
- Uses the owner's compute stream. **15 launches per 64-token chunk**, including
  a padded tail; one row remains one existing launch. No allocation, copy or
  synchronization inside submission. Counters count successful submissions, not
  proof of completion or rollback after a partial launch failure.
- Retain state/output/workspace through stream completion/retirement. Reuse on the
  same ordered stream is allowed; cross-stream/owner reuse needs explicit existing
  fencing. An error after partial submission can leave state/output/scratch
  modified: quarantine all of them on **Unknown** completion. The borrow wrapper
  does not itself prove completion or reset poison. Existing allocator retirement
  and owner lifetime machinery remain unchanged. Mid-launch fault/quarantine
  injection was not added or claimed tested in this bounded slice.

Model integration must explicitly budget/own scratch, select this backend method
for multi-row GDN, and include workspace in pending/Unknown state custody. No
model/composition file or default dispatch was switched here. Full-model and
broader performance acceptance remain integrator work.

The 2026-10-07 launch-fusion implementation and original preparation are in
`CHUNK_FUSION.md`. Earlier 19-launch GPU results below are historical evidence,
not a performance claim for the new candidate. Fusion exact GPU acceptance
and the subsequent explicit one-row prefill gates are recorded in `PREFILL_ONE.md`.

## Arithmetic

Each chunk pads Q/K/V/g/beta to64, preserving BF16 L2 and beta boundaries before
F32 promotion. It computes prefix g with a Sklansky tree, decay matrices, K_beta
and V_beta, a lower triangular transform, transformed values/decayed keys, output
and F32 state update. Chunks advance chronologically with the previous F32 S.

The internal tiled SIMT matmul uses **IEEE F32 FMA**, not TF32x3 or BF16 products.
Every elementwise materialization boundary uses separate RN multiply/add/subtract.
There is no GEMM beta epilogue that contracts the state-update boundary. Matrix
products are bounded to the chunk panels; no library-side hidden workspace.

The triangular transform snapshots each row before updating it, keeps its
cross-row dependency within one block, and uses four interleaved F32 accumulators
for the materialized product reduction. This matches the pinned HF non-fast-axis
`Reduce.cuh` tree for r<64, rather than a serial sum. Each result is shape-general:
there are no fixture inputs, token/head/index corrections or substituted outputs
in the implementation.

### Exceptional values and overflow

Numerical validity requires finite inputs/state **and bounded intermediates at
all BF16/F32 materialization boundaries**, including normalization, gate
exponentials, prefix sums, triangular/matrix reductions, state updates and BF16
output rounding. Shape, extent and ownership admission do not inspect values.
There is **no universal finite-result guarantee for finite operands**.

For example, finite BF16 `A_log=100` with `a=dt_bias=0` overflows F32
`exp(A_log)`, giving `g=-Inf`; subtracting two such prefixes yields NaN.
Even individually finite gates can overflow a chunk prefix sum. Padding zeros
are not a numerical firewall: `0 * NaN` remains NaN and can contaminate output
and carried state. NaN/Inf operands and overflowing intermediates are not
clamped, sanitized or reported as a numerical-error status. Successful launch
submission says nothing about numerical validity, and exceptional-value
bit parity with HF or another toolchain is not promised. Underflow/rounding can
also lose information; bounded intermediates are not a universal accuracy proof.
One-row delegation retains the existing operator's arithmetic unchanged.

Dedicated CPU tests demonstrate the F32 overflow/NaN mechanisms only. They do
not execute CUDA, reproduce its transcendental/reduction implementation, or
establish exceptional-value GPU output/state behavior.

## Evidence and resolved dense-case defect

Final native fixture/evidence:

- `target/validation/gdn-chunk-backend-release-20261006`
- `target/validation/gdn-chunk-backend-release-run-20261006`
- `target/validation/gdn-chunk-rust-release-run-20261006`
- `target/validation/gdn-chunk-dense-stage-fixed-20261006`

The actual capital layer0 five-row inputs/reference remain pinned to the retained
independent capture and its original HF stage payload. New synthetic references
are observations of unmodified Torch2.13cu129/Transformers5.5.3, never native output.

**Actual:** 20,480/20,480 BF16 output words and 524,288/524,288 F32 state words exact.
Every compared intermediate and all eight matmul results are exact, both in the
pipeline and isolated from original HF operands. The prior affected output
flat10476 is now `0x33f2`, not the chronological scan's `0x33f1`.

**Synthetic:** seven structured shapes (rows2/5/63/64/65/67/129, grouped heads,
Dk1/4/16/17/64, Dv3/5/7/9) plus dense rows5/65 with nonzero decay and non-BF16 F32
initial state. All nine cases have bit-exact output and final F32 state. Tails,
uneven dimensions, a one-token final chunk and three sequential chunks are covered.
Across actual and synthetic cases: 27,269 output and 526,528 state values exact.
These cases do not prove arbitrary dimensions/inputs or other GPU/toolchain parity.

An initial serial triangular reduction passed actual rows5 but failed dense65 in
4/288 final-state words (max1.862645149230957e-9), with all BF16 outputs exact.
Independent original-function observation localized the first discrepancy to
chunk0 triangular row6; input products, prefix, decay and pre-transform triangle
were exact. Replacing the serial sum with HF's four-accumulator order eliminated
that discrepancy. **All 30 observed dense65 stages are now exact**, including
both chunks' transformed values/keys and nonzero-state projections. Earlier
failure receipts/stages are retained; no tolerance/reference was changed.

The dense-stage gate independently requires
`gdn-chunk-dense-stage-fixed-20261006/dense-stage-comparison.json`, pinned by
SHA-256 `d3d4b8375eb2707a864f33b6962d4c979edeb9a8a85a050336e3ee215c529291`.
It requires exactly the 30 named stages with their expected element counts,
zero unequal values/max absolute error, empty mismatch indices, and unchanged
original HF output/state. Final output/state equality alone cannot set
`dense_observations_exact`. New gate summaries record receipt path/hash/count;
CPU checks also validate the receipt directly, including for historical summaries
that lack those fields. This is retained fixed-input evidence, not a fresh GPU
stage observation for a new run or changed kernel.

Rust GPU gates additionally passed: missing/short/mismatched/foreign-owner scratch
and foreign inputs reject with zero launches; valid actual submission has19
launches and zero allocation attempts, H2D/D2H copies or stream-wide syncs. The
single-row method delegates exactly to the existing path. Native tests check both
allocation-end canaries, input immutability, observed/unobserved equality, and
invalid ABI/budget/overflow/alias descriptors before any CUDA call.

sm86 resource reports: matmul36 registers, 2,048 shared bytes; triangular39
registers, 16,640 shared bytes; zero stack/spill stores/spill loads. **Chunk event
performance has not been measured**. These counts are not a performance claim.

## Reproduction (GPU owner only)

CPU-only preparation into a fresh directory:

```sh
. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
CUDA_VISIBLE_DEVICES='' /opt/conda310/bin/python -B \
  crates/ferrule-backend/tests/recurrent_bf16/chunk_gate.py prepare \
  target/validation/gdn-chunk-backend-rerun-20261006
```

Bounded GPU gate, only after acquiring exclusive ownership:

```sh
. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
/opt/conda310/bin/python -B \
  crates/ferrule-backend/tests/recurrent_bf16/chunk_gate.py run \
  target/validation/gdn-chunk-backend-rerun-20261006 \
  --evidence target/validation/gdn-chunk-backend-rerun-receipts-20261006
```

Existing authenticated supervisor: HF child120s, each fixed native case60s,
plus bounded drain and three all-eight-GPU idle samples. No model/full40.

CPU evidence checks for the final retained run (source pins intentionally fail
if harness or production sources have changed; do not rewrite historical pins
to turn that into a pass):

```sh
CUDA_VISIBLE_DEVICES='' /opt/conda310/bin/python -B \
  crates/ferrule-backend/tests/recurrent_bf16/test_chunk_backend.py \
  --prepared target/validation/gdn-chunk-backend-release-20261006 \
  --evidence target/validation/gdn-chunk-backend-release-run-20261006
CUDA_VISIBLE_DEVICES='' cargo test --locked -p ferrule-backend \
  --no-default-features --test recurrent_bf16_chunk_plan
```

To run only the dedicated receipt rejection/acceptance tests and CPU exceptional
arithmetic examples after editing the harness (without claiming historical
source-pin acceptance):

```sh
CUDA_VISIBLE_DEVICES='' /opt/conda310/bin/python -B \
  crates/ferrule-backend/tests/recurrent_bf16/test_chunk_backend.py \
  --prepared target/validation/gdn-chunk-backend-release-20261006 \
  --evidence target/validation/gdn-chunk-backend-release-run-20261006 \
  DenseStageReceiptTests ExceptionalArithmeticPolicyTests
```

These tests do not supply fresh GPU evidence. Still untested here: GPU
exceptional-value results, instrumented/partial-failure launch counts, changed-input
cross-call state carry with reused scratch, Unknown-completion quarantine, and
long dense nonzero-decay sequences or near-limit dimensions.

The dedicated ignored Rust test target is `cuda_recurrent_bf16_chunk`; supply
`FERRULE_GDN_CHUNK_FIXTURE` pointing at prepared `actual/`. Run only those two
ignored tests, serially, under the authenticated supervisor. The final local
wrapper is `target/validation/gdn-chunk-backend-release-20261006/supervise-rust.py`
(90s), which records source/binary/library pins and release. Its positional argument
is a fresh evidence directory. Rebuild before using it after source changes.

The native CUDA12.4 toolchain lacked the unversioned `libcublas.so` linker name;
Rust linking used an evidence-local symlink to the existing authenticated
`libcublas.so.12.4.5.8`, via `LIBRARY_PATH` (no package installation/toolchain edit).
The chunk kernel itself does not call cuBLAS.

## Continuation follow-up: CPU-only fixes, GPU validation pending

The broader continuation owner subsequently found four failing cases; the earlier
actual/nine-case results above are historical, not universal parity. See
`target/validation/chunk-continuation-performance-gpu-20261006/BACKEND-HANDOFF.json`.
This follow-up ran with GPUs hidden while the backend GemmEx owner was active.
No GPU test, profiler, device query, or quiescence claim was made in this cycle.

Two changes are confined to `recurrent_bf16_chunk.cuh`:

1. **Prefix geometry:** pinned Torch `ScanUtils.cuh` selects the innermost scan
   window using the complete number of rows `Hv * ceil(sequence_rows/64)`, not
   merely one resident chunk. At Hv32/rows129 this is96: x-thread count16,
   two32-value windows. The total of the first window is added to the first
   element of the second **before** its Sklansky scan. CPU F32 replay reproduces
   the old native mismatch counts378/372/0 by chunk; the corrected order matches
   all6144 frozen HF prefix words. The new host selector follows the pinned
   uint32 geometry calculation and passes the window as an internal kernel arg.
   At Hv32/rows5 or64 the window remains64. Geometry is evaluated separately for
   each invocation; carry between scan windows is not SSM/inter-chunk state.
2. **Query scale:** HF computes `1 / (Dk ** 0.5)` as a Python double, then converts
   the scalar to F32 for tensor multiplication. Native `1.0f / sqrtf(Dk)` can
   double-round. At Dk511 the old/new scale bits are `0x3d353246`/`0x3d353245`.
   Compute the double scalar on the host once, convert to F32, and pass it to the
   preparation kernel. All512 admitted scalar values match Python's expression
   in the host-helper test. The change predicts zero differences for all65408
   frozen query values versus the old1912 differences. The capture omitted HF
   qnorm: the CPU test reconstructs BF16 qnorm candidates, first requiring exact
   reproduction of **every** native old-scaled word. It does not label those
   reconstructed candidates as newly observed HF normalization.

No ABI, FFI, plan, scratch size, launch count, admission limit, single-row policy,
state dtype, ownership or completion-policy changes. Existing `recurrent_bf16.cuh`
and GemmEx/provider code are untouched. These fixes are **not yet GPU-validated**.
Pinned Torch's special single-scan-row deterministic CUB path is distinct from
`ScanUtils` multi-row dispatch; this change retains its previous native behavior
and does not establish universal singleton-scan parity from two-token captures.

### Remaining K512 matrix blocker

On both Dk512 cases, exact HF K_beta and normalized K operands reproduce the
native result under CPU IEEE-F32 sequential FMA, but differ from HF in4/4096
matmul0 results. Max errors are3.5762786865234375e-7 (Dv511) and
2.384185791015625e-7 (Dv512). Separately rounded multiply/add and a final F64-dot
cast do not explain all HF results either. Tested contiguous/interleaved2/4/8/16/32
accumulator candidates do not reproduce both captures; no candidate justifies a
production correction. **Matmul is unchanged.**

The recorded contract is F32 `[1,1,1,64,512] @ [1,1,1,512,64]`, transposed K,
TF32 disabled, precision `highest`, deterministic Torch2.13cu129. It does not pin
cuBLAS's internal kernel/algorithm or reduction tree. The next bounded owned run
profiles only these two exact operand products and verifies their original HF
results before and after profiling. No vendor kernel or split-K policy is inferred
from a few matching outputs. Any future BLAS handle/workspace/FFI integration must
be coordinated with the GemmEx owner first; none was introduced here.

### Prepared gates (do not run without new GPU ownership)

CPU evidence: `target/validation/gdn-chunk-math-cpu-20261006/cpu-evidence.json`.
Five CPU math/source tests, eight continuation fixture/contract tests and four
Rust plan tests passed. CUDA12.4/sm86 standalone builds and `cargo check --features
cuda` passed with GPUs hidden. New continuation inputs are verified byte-identical
to the originals, including carried calls/reset seeds. Original independent HF
expectations are read-only and are not regenerated by this runner.

After explicit GPU ownership, run the existing actual/5/64/tail baseline first:

```sh
. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
/opt/conda310/bin/python -B \
  crates/ferrule-backend/tests/recurrent_bf16/chunk_gate.py run \
  target/validation/gdn-chunk-math-cpu-20261006/baseline-prepared \
  --evidence target/validation/gdn-chunk-math-baseline-gpu-20261006
```

Then the retained-reference continuation/reset gates and exact HF matrix profiles:

```sh
. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
/opt/conda310/bin/python -B \
  crates/ferrule-backend/tests/recurrent_bf16/chunk_math_regression.py run \
  target/validation/gdn-chunk-math-cpu-20261006/continuation-prepared \
  --evidence target/validation/gdn-chunk-math-continuation-gpu-20261006 \
  --execute-cuda
```

Every native case/HF matrix profiler is authenticated and bounded to120s; cleanup
and three final idle samples are additional. This is not a timing/performance run.
It intentionally remains a failing exact gate until the K512 blocker is resolved;
all numerical differences are saved without tolerances or output substitution.
Preparation source pins fail closed if concurrent backend work changes a pinned
file. Reprepare into a fresh directory rather than editing a stale manifest.

## GPU-verified continuation fixes and measured timing

The preceding CPU-only/pending section is now superseded by fresh authenticated
CUDA12.4/sm86 evidence in `target/validation/gdn-chunk-splitk-20261006/FINAL.json`.
The original independent HF payloads/tolerances and all failure receipts remain
unchanged. No model/default switch, GemmEx edit, new BLAS handle/workspace provider,
public profile, or change to the existing `recurrent_bf16.cuh` was made.

### Prefix and scale verification

With the host-double query scalar and full-tensor cumsum geometry fixes:

- Dk511/Dv512 first-call/carry/reset output and F32 state are exact. All20 observed
  first-call intermediates, including all65408 scaled-query values, are exact.
- Hv32/Dk128/Dv128 dense129 output528384/528384 and state524288/524288 are exact,
  including the separate reset. All60 captured stages across its three chunks
  are exact. This independently validates the32-value-window prefix correction.
- Existing actual5/dense64 and the10-case original backend gate remain exact.

### K512 root causes and native fix

The pinned HF matrix profile (Torch2.13cu129, `allow_tf32=False`, precision highest,
`:4096:8`) identified two different reduction implementations:

1. Single-batch `[64,512] @ [64,512]^T`: kernel
   `ampere_sgemm_32x32_sliced1x4_tn`, grid `[2,2,8]`, block `[128,1,1]`, followed by
   a separate split-K reduction. The retained CUDA12.9 library SASS shows four
   warp slices with eight contiguous K values, stepping every32, and the pairwise
   shared-memory sum `(slice0+slice1)+(slice2+slice3)`. Eight contiguous K64
   partition sums are then added in ascending order.
2. Single-batch `[64,512] @ [512,511 or512]` state projections: kernel
   `cutlass_80_simt_sgemm_128x32_8x5_nn_align1`, grid `[8,1,8]`, followed by split-K
   reduction. Each K64 partition is a sequential F32 FMA sum, then the eight
   partitions are added in ascending order. It has no four-slice inner reduction.

The chunk's existing SIMT matmul now emulates these two arithmetic orders locally.
Every output uses the same shape-selected algorithm; there is no token/head/index
correction or output replacement. Partials remain in registers, with no new
workspace or launches. Other transpose/shape/batch regimes retain the original
SIMT order. The internal call also receives the **reference logical batch count**:
precomputed products use `Hv*chunks`, per-chunk state/output products use `Hv`.
This avoids treating streamed scratch as proof that HF's original product had
only one batch. Two-/three-batch independent probes pass with the ordinary path.

Evidence is not just a fit to four retained values: `chunk_splitk.py` runs the
**production** matrix function on14 cases (retained matrices, dense signed/positive,
wide-magnitude/cancelling inputs, both511/512 state widths, and batches2/3), all
**241472 F32 matrix elements exact** against independently executed pinned HF.
Kernel profiles accompany those expectations. No TF32 or changed oracle.

The intermediate investigation is retained: fixing only QK matmul removed its
error but left state projection discrepancies; the second observed reduction
contract resolved them. Final first-call stages are exact20/20 for each near-limit
case and60/60 for dense129: **120/120 stages exact**.

### Final correctness and lifecycle gates

`final-run/comparison.json`: **7/7 continuation/reset cases exact**, comparing
1641039 BF16-expanded output values and6288000 F32 state values. Includes changed
5->65->129 inputs, all three near-limit geometries, actual5, dense64 and dense129.
The existing dirty/fresh scratch, carried state, no inter-call host fence, input
immutability and canary checks also pass. No numerical failure is suppressed to
allow timing.

`baseline-run`: original10-case output/state/intermediate gates pass.
`rust-run`: both GPU tests pass, including19-launch resident submission and
missing/short/mismatched/foreign-owner scratch rejection. No launch allocation,
copy or synchronization; one-row delegation remains the existing path.

CPU checks passed:8 continuation tests,5 new split-K evidence tests,4 Rust plan
tests and18 historical-release evidence tests. The historical test intentionally
rejects a new preparation manifest; it was rerun against its pinned original
release rather than changing its trust anchor. CUDA check/build and formatting
checks pass. Mid-launch fault/quarantine injection is still not claimed tested.

### Actual measured scan/chunk event timings

`final-run/performance.json` contains raw paired samples, numerical errors,
medians/p95s and authenticated receipts. Dk128/Dv128/Hv32; four warmups per variant,
20 alternating AB/BA paired event samples; same addresses and seed state restored
outside each start event. Full output/state and guards are checked outside timing.

| Rows / input | Scan median (us) | Chunk median (us) | Scan p95 (us) | Chunk p95 (us) |
| --- | ---: | ---: | ---: | ---: |
| 5 / actual capital | 268.288 | 443.904 | 274.893 | 451.789 |
| 64 / independent dense | 2936.320 | 531.456 | 3055.207 | 538.726 |
| 129 / independent dense | 5783.040 | 1504.260 | 6045.385 | 1509.380 |

**Actual5 chunk latency is65.5% higher.** The longer rows have lower measured chunk
latency, but these are **different numerical profiles**, not equal-quality speedup
claims: scan output/state unequal counts are1/302175,225/494954,400/488773 for
rows5/64/129 respectively. Every timed chunk output/state is HF-exact. No throughput
or serving projection, no full40. The current matmul uses39 registers and2048
shared bytes; triangular39 registers/16640 shared bytes; zero stack/spill traffic
in the standalone sm86 build. Scratch/ABI/19-launch-per-chunk budget is unchanged.

### Bounded rerun

Fresh preparation is required after any pinned source changes. Under exclusive
GPU ownership, the final prepared gate can be reproduced with a fresh evidence dir:

```sh
. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
/opt/conda310/bin/python -B \
  crates/ferrule-backend/tests/recurrent_bf16/chunk_math_regression.py run \
  target/validation/gdn-chunk-splitk-20261006/final-prepared \
  --evidence target/validation/gdn-chunk-splitk-rerun-20261006 \
  --execute-cuda --performance
```

Each native/profile/timing child is bounded to120s under the existing authenticated
supervisor. Timing runs only after **all** continuation cases pass exactly. Native
CUDA12.4 and pinned HF CUDA12.9 run in separate processes. Final all-GPU idle/drain
receipts accompany the run.

No known failure remains in these gates. This does **not** prove arbitrary shapes,
BLAS dispatch regimes, GPUs/toolchains, singleton CUB scan geometry, or full-model
parity. Exact shape/batch dispatch is intentionally documented rather than
presented as universal cuBLAS emulation; admission was not lowered to hide cases.
