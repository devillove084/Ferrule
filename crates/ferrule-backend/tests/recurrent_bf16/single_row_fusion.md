# Single-row GDN pass fusion

## Authenticated GPU results (2026-10-06)

Fresh evidence: `target/validation/gdn-single-row-fusion-gpu-20261006/FINAL.json`.
The originally prepared source and binary/fixture hashes all remained valid;
no kernel or reference changes were needed. CUDA12.4/sm86 execution on RTX3090
used authenticated subreaper custody, bounded commands, and per-run idle checks.

- Actual frozen output **4096/4096 BF16 words exact** and final state
  **524288/524288 F32 words exact** to independent HF; output2603 remains3911.
- Production output/state also exactly match the pinned pre-fusion observer.
  All observer stages match HF. These traces are explicitly from that observer,
  **not newly instrumented fused internals**; the fused output/state check and
  CPU expression-DAG proof provide the separate connection. No old instrument
  generator was run against incompatible new source anchors.
- **88/88 selected-shape full-update cases**,21342720 state/output words, pass
  against exact unfused with guard and input-immutability checks.
- **107/107 multirow/excluded stability cases**, including10 below-threshold
  cases, and **55/55 reduction cases** pass.
- Both fenced benchmark comparisons pass complete output/state, input and
  guard checks for every timed sample, for rows1/5/32.

Contemporary matched median latencies in microseconds, two independently
supervised passes (suite execution order reversed on the second pass):

| Comparison | Rows | Baseline medians, pass1 / pass2 | Fused medians, pass1 / pass2 | Median paired latency change, pass1 / pass2 |
|---|---:|---:|---:|---:|
| Exact unfused vs fused | 1 | 77.952 / 77.952 | 67.072 / 66.944 | -14.04% / -14.09% |
| Exact unfused vs fused | 5 | 268.608 / 270.208 | 255.040 / 261.568 | -3.81% / -3.05% |
| Exact unfused vs fused | 32 | 1350.720 / 1355.328 | 1254.144 / 1250.624 | -6.83% / -7.71% |
| Archived scalar vs fused | 1 | 60.288 / 60.544 | 67.072 / 67.136 | +11.28% / +10.58% |
| Archived scalar vs fused | 5 | 263.296 / 257.920 | 263.040 / 258.112 | -0.10% / +0.02% |
| Archived scalar vs fused | 32 | 1258.944 / 1255.232 | 1258.880 / 1256.256 | -0.06% / +0.05% |

Each pass has30 AB/BA paired samples per shape,8 reset-fenced events per sample,
20 warmups per variant, identical buffer addresses, and reset/checking outside
events. The paired change is the median of per-pair ratios, not the ratio of
separate medians. Fused beats exact-unfused in all60 single-row pairs, but is
slower than archived scalar in all60 pairs. Row5 has substantially wider
variation (roughly220–275us); all raw event samples, ranges, MADs, order-split
ratios and boundary temperature/clock telemetry are retained. No fixed-clock
or continuous-telemetry assertion is made.

Runtime eager-kernel attributes confirm45->40 registers, zero local/shared
bytes. PTXAS also confirms the selected scalar scan uses40 registers and all
production kernels have zero spills/stack. The scalar scan's1184 SASS
instruction texts remain identical to archived scalar. The benchmark's shared
resource label queries the eager kernel; scan resources are established by
PTXAS, not misattributed from that runtime query.

Historical51.968/66.944us measurements are not a matched comparator for this
run. The residual single-row cost relative to archived scalar remains visible.
These are absolute bounded GDN operator timings, **not model throughput,
serving, full40, or chunk-backend performance claims**.

## Scope and implementation

Only production `native/cuda/core/recurrent_bf16.cuh` changes. Dedicated new
`single_row_*` tests reuse the shared suites read-only; no provider/FFI, chunk
kernel/test, shared benchmark, profile, or model changes.

The exact unfused kernel reads the full state four times: decay, memory
reduction, correction update, and dot reduction. `column_update_sum4<Update>`
fuses each update with the corresponding reduction, retaining the just-written
F32 value for the product. There are now two state-read passes and still two
state-store passes. No extra scratch, allocation, launch, or synchronization.

Arithmetic is unchanged:

- The existing shape predicate remains `rows==1`, `64<=Dk<=512`, `Dv%4==0`,
  `Hv*Dv>=128`; one-row prefill is still included.
- Each d has a unique lane/accumulator/iteration: `d = lane + 4*j + 16*i`.
  Its state update can move to that visit because each thread owns the column
  and all other dimensions are independent. Existing non-aliasing requirements
  remain necessary.
- Each accumulator starts at +0 and retains increasing i order. Each lane uses
  `(((acc0+acc1)+acc2)+acc3)`; lanes merge `(p0+p2)+(p1+p3)`.
- Decay, correction, products, and additions retain explicit `__fmul_rn`,
  `__fsub_rn`, `__fadd_rn`. Normalization and BF16 boundaries are unchanged,
  including the memory product's scale-by-one operation. No FMA substitution.
- The existing public entry point launches exactly one kernel on the caller's
  stream. Eligible shapes use `delta`; all other shapes use `delta_scan`.
  The latter's body is identical to the archived original scalar scan, apart
  from its internal kernel name. This separates register allocation without
  adding an exposed precision/profile variant or changing launch-status policy.

The original `column_sum4` stays unchanged for the existing reduction matrix
and archived observed-stage regression. It is not used by the fused production
path. No claim is made about unvalidated aliasing inputs or universal HF
transcendental/reduction parity.

## Completed CPU checks

Evidence: `target/validation/gdn-single-row-fusion-final-20261006/PREPARED.json`.
Its immutable preparation status is **CPU_PREPARED_GPU_NOT_RUN**; the later GPU
receipts above are separate. All preparation compilation/host executions used
`CUDA_VISIBLE_DEVICES=''` and `NVIDIA_VISIBLE_DEVICES=void`; no CUDA device query
or nvidia-smi was issued during that CPU-only preparation phase.

1. **Ordered expression DAG:** `single_row_order.cpp` compiles the extracted
   production helper bodies with symbolic operators rather than floats. For
   every Dk=64..512 and strides1/4/7 (1,347 cases), memory, correction, decayed
   and updated state, output dot, BF16 boundary, and untouched guards have
   identical ordered expression nodes. Leaves and operator operands are not
   reordered or algebraically simplified. It checks 387,936 state words per
   stage. Deliberate stride, merge-order, and missing-BF16 mutations all fail.
2. **Frozen numeric CPU check:** `single_row_numeric.py` uses separately rounded
   NumPy F32 operations on the retained HF stage operands, without Torch/CUDA.
   Memory4096, decayed state524288, correction4096, updated state524288,
   dot4096, and output4096 all have zero unequal bit patterns. Norms/gates are
   retained stage operands, not newly claimed CPU/CUDA transcendental parity.
3. **CUDA12.4.99/sm86 PTXAS:** exact unfused delta45 registers; fused delta40;
   isolated scalar scan40. All three: zero stack bytes, zero spill stores,
   zero spill loads. The compiled instrumented observer has68 registers and
   zero spills; it is test-only, never the production timing target.
4. **Excluded-path code generation:** old scalar vs isolated scalar has all
   1,184 SASS instruction texts identical. See
   `target/validation/gdn-single-row-fusion-20261006/scalar-sass-comparison.json`.
5. Existing no-default-feature recurrent tests: 3 passed. The validation target
   has no tests enabled without CUDA; this is not GPU validation.

Source diff against the pinned exact unfused header:
`target/validation/gdn-single-row-fusion-20261006/implementation.diff`.
Its SHA256 is `15a6e71ea871e2bdf37be48fa65f3afcfdb874c2e54d3847f430ee87b3a4b844`.
Final production source SHA256:
`14695c068c9267ac2447a91f2726cbc3611e9633ba70507da5de0727c439e8b2`.

The CPU/compiler evidence alone did not establish runtime improvement. The
subsequent authenticated measurements are reported separately above.

## Prepared GPU gates — only after an explicit grant

`single_row_prepare.py` generates source/binary/fixture pins and ready command
plans. It never runs CUDA and refuses to reuse its output directory. Reproduce
with a fresh output path and the retained pinned pre-fusion header:

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void /opt/conda310/bin/python -B   crates/ferrule-backend/tests/recurrent_bf16/single_row_prepare.py   target/validation/gdn-single-row-fusion-reprepare-20261006   --exact-unfused-header target/validation/gdn-single-row-fusion-20261006/before.cuh
```

Already compiled under `target/validation/gdn-single-row-fusion-final-20261006`:

- `actual-plan.json`: existing `actual_replay.cu`, original frozen input and
  pinned unfused observer, checks complete output/state and canaries. Follow
  with `actual-compare-plan.json` (`--require-exact`) for all frozen stages.
  The observer is deliberately the pinned pre-fusion instrumented kernel. Do
  **not** use the old `actual_replay.py instrument` text-anchor generator on
  the new fused kernel; its decay-store anchor no longer exists. Production
  output/state comparison and symbolic dependency checks are separate evidence.
- `synthetic-plan.json`: 88 full-update selected-shape cases vs exact unfused,
  Dk64/65/67/127/128/129/255/256/257/511/512; Hk/Hv/Dv1/1/128,
  1/4/32,2/32/4,16/32/128; zero and nonzero initial state, non-neutral gates,
  full BF16 output/F32 state, input immutability, both-end guards.
- `exact-performance-plan.json`: fenced AB/BA protocol with identical addresses
  and fresh-state resets, 20 warmups,30 rounds,8 events per sample. Here the
  shared labels **old=exact unfused**, **new=fused**; both row1 variants use the
  same independently pinned HF oracle, never replacement expected outputs.
- `matrix-plan.json`: unchanged 55-case independent HF reduction matrix. This
  validates the retained reduction helper, not the new full-update path alone.
- `legacy/`: existing original-scalar-vs-candidate benchmark and107-case
  multirow/excluded stability suite, including10 below-threshold cases.

Run the ready plans through the current GPU owner's authenticated supervisor,
checking PREPARED source/artifact hashes immediately before execution. Require
matching launch/helper identities, proven-drained custody, no timeouts/errors,
and three idle observations before release. Plans carry60/90/120-second limits;
they are not permission to take another agent's GPU.

The existing benchmark/stability runner is directly ready after ownership is
explicitly granted and all devices are released:

```sh
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/admin/wxw403883/anaconda3/envs/tao/lib:/usr/lib64   /opt/conda310/bin/python -B crates/ferrule-backend/tests/recurrent_bf16/performance.py run   target/validation/gdn-single-row-fusion-final-20261006/legacy   --evidence target/validation/gdn-single-row-fusion-gpu-20261006 --execute-cuda
```

It authenticates libraries, bounds tests, records timings and checks release.
For native actual replay, the current owner can submit `actual-plan.json` and
then `actual-compare-plan.json`; no fixture export or full40 is necessary.
Run exact frozen and synthetic gates before interpreting performance results.
