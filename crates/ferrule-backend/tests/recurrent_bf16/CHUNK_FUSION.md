# Validated follow-up

All prepared fusion correctness gates passed under authenticated CUDA12.4 before
the explicit one-row prefill API change. Final-source fused panels and120 frozen
HF stages also pass. See `PREFILL_ONE.md` and
`target/validation/gdn-prefill-one-20261007/FINAL.json`. No fused/unfused timing
claim. The original CPU preparation record follows unchanged.

# Chunk launch fusion — implemented, GPU validation pending

The authorized 19-to-15 launch candidate is implemented in the native chunk
header and Rust launch plan. This cycle is CPU-only with GPUs hidden; another
agent owns the GPUs. No new numerical or performance pass is claimed. Prefix,
query scaling, triangular and all matrix/split-K arithmetic remain unchanged.

Motivation: measured actual5 chunk443.904us versus chronological scan268.288us,
with19 chunk launches. This does not establish that launch overhead accounts for
the difference: normalization, padded64-row work and the triangular transform also
cost time. Fewer launches are a plausible experiment, **not a latency prediction**.
Scan and chunk differ numerically; the scan is not an equal-quality replacement.

## Archived pre-fusion graph: 19 launches per chunk

All arrows are same-stream launch order. `MM0..MM7` retain their existing IEEE-F32
FMA/reduction, transpose, logical-batch and output-materialization contracts.

```text
 1 prepare
 2 prefix
 3 decay
 4 exponent_keys(false): ke=kb*exp(prefix), qe=q*exp(prefix)
 5 MM0: kb @ k.T -> tri
 6 mask_matrix(true): tri=-tri*decay, strict lower triangle
 7 triangular
 8 MM1: tri @ vb -> value
 9 MM2: tri @ ke -> kc
10 MM3: q @ k.T -> attn
11 mask_matrix(false): attn=attn*decay, lower triangle including diagonal
12 MM4: kc @ S -> vp
13 delta_value: vn=value-vp
14 MM5: qe @ S -> inter
15 MM6: attn @ vn -> intra
16 output: BF16(inter+intra)
17 exponent_keys(true): ke=k*exp(prefix_end-prefix)
18 MM7: ke.T @ vn -> state_product
19 update_state: S=S*exp(prefix_end)+state_product
```

## Implemented candidate: 15 launches, matrix kernels unchanged

```text
 1 prepare
 2 prefix
 3 decay_and_initial_exponents                 [old3 + old4]
 4 MM0
 5 mask_matrix(true)
 6 triangular
 7 MM1
 8 MM2
 9 MM3
10 MM4
11 attention_delta_and_end_keys                [old11 + old13 + old17]
12 MM5
13 MM6
14 MM7
15 output_and_state                            [old16 + old19]
```

These are ordinary barrier-free elementwise fusion kernels, not cooperative
multi-stage launches. The existing eight matrix kernels and triangular/prefix
kernels remain separate. Same-stream completion between launches supplies every
cross-block dependency; no device-wide synchronization or atomic protocol is added.

### Fusion safety and extents

Let `nm=Hv*64*64`, `nk=Hv*64*Dk`, `nv=Hv*64*Dv`, `ns=Hv*Dk*Dv`.
Use the existing256-thread elementwise block size; each branch independently
bounds its index. Never early-return after just one branch's bounds/tail check.

| New kernel | Launch extent | Reads | Writes | Why safe |
| --- | --- | --- | --- | --- |
| `decay_and_initial_exponents` | `max(nm,nk)` | completed prefix, kb, q | decay, ke, qe | Distinct panels, no branch consumes another branch's result. All consumers are later launches. |
| `attention_delta_and_end_keys` | `max(nm,nv,nk)` | raw attn, decay, value, completed vp, k, prefix | attn, vn, ke | Attention mask may move after MM4: MM4 does not read attn. MM2 has completed before ke is overwritten. Old ke is dead; kc and qe are distinct and remain intact. MM5/6/7 are later consumers. |
| `output_and_state` | `max(nv,ns)` | inter, intra, old S, prefix_end, state_product | BF16 output, F32 S | All old-S readers (MM4/MM5) and MM7 have completed. Output and S are validated disjoint. MM7 does not read BF16 output, so delaying output is safe. Next chunk cannot start until this launch completes. |

For small Dk or uneven Dv, these extents differ; all must be handled. The output
branch keeps the real-token tail predicate, while the state branch updates **all**
SSM words regardless of the output-tail predicate. No new panel aliasing or memory
reuse is introduced. `w.ke` is reused at exactly its existing semantic lifetime
boundary, only earlier after its last reader.

### Preserve every arithmetic boundary

Copy the current elementwise expressions verbatim:

- Prefix differences use `__fsub_rn` before `expf`.
- Products use `__fmul_rn`; sums/subtractions use `__fadd_rn`/`__fsub_rn`.
- State update remains separately rounded multiply then add, **not FMA**.
- Output remains separately rounded F32 add then `bf16_rne`.
- Preserve attention upper-triangle+0 and the existing strict-lower triangular
  mask's-0 behavior. Do not simplify zero/identity cases based on fixture data.
- Preserve BF16 normalization/beta, F32 state, all logical batch counts, and
  split-K rules. No changing matmul epilogues to achieve the fusion.

## Resource/lifecycle contract

Unchanged: native public ABI, exact typed operand extents, caller-owned scratch
allocation/layout,17,850,368 logical scratch bytes at Qwen geometry, disjoint/owner
validation, zero hot-path allocations/copies/synchronizations, and one-row dispatch.
No new shared memory or scratch panels are required by these barrier-free kernels.
Register use, spills, occupancy and event timing are **not proven unchanged**;
merged kernels may require more registers or suffer branch/load inefficiency.

Changed: accepted launch accounting becomes `15*ceil(rows/64)` (one row remains1).
A standalone header-only candidate is possible for experimentation, but production
integration must also update `Bf16ChunkPlan::launch_count`, native
`launches_per_chunk`, and the dedicated count assertions (19/38/57 ->15/30/45).
This metadata change must be coordinated; do not silently ship a header-only change
while the owner reports19. Budget/layout requirements themselves do not change.

Retain all operands through existing stream retirement; partial submission is not
rollback. Output is published one launch later in the graph but remains unavailable
to unordered readers until completion. Unknown completion still quarantines both
state and scratch; removing launch boundaries must not create early retirement.

## Minimal acceptance experiment after ownership/source clearance

1. Keep the current19-launch kernel as a **test-only** archived baseline. Build
   original and candidate under CUDA12.4/sm86 from the same input contracts.
2. Dedicated hooks compare all fused outputs bitwise against unfused elementwise
   outputs on the existing frozen operands: decay/ke/qe, masked attn/vn/end-ke, and
   BF16 output/F32 state. Include signed-zero masks, padded tails, unequal nm/nk/nv/
   ns, nonzero state and dirty scratch. Store hook observations only after the fused
   kernel completes; no production synchronization or hidden trace allocation.
3. Require all original10 backend cases,7 continuation/reset cases,120 independent
   intermediate stages and14 matrix probes to remain exact. Reuse immutable HF
   references. Check both-end guards, input immutability, owner/alias/budget rejects,
   same scratch/state addresses, and no host fence between carried calls.
4. Rust resource test:15 launches per chunk, unchanged zero allocation/copy/sync
   counters, correct15/30/45 multi-chunk counts and unchanged one-row result/count.
5. Only after exact gates, repeat the existing matched event test for actual5,
   dense64,dense129: compare unfused chunk versus fused chunk, same seed/addresses,
   alternating order, full checks outside timing. This is now an equal-result
   comparison. Record registers/spills too; accept only a measured improvement,
   with no throughput projection. No full-model run or profiling detour required.

## CPU implementation and prepared gates (2026-10-07)

`chunk_unfused.cuh` is the immutable test-only pre-fusion production header,
SHA256 `2182f58b6f5543d681dc6fbd1873bb0d4de1e27a3b4e265f85af001af1c5d0f7`.
It includes all final prefix/scaling/split-K fixes. Production never includes it.
`chunk_fusion.py` generates a private namespace copy, builds both variants in the
same CUDA12.4/sm86 TU, and prepares existing dedicated gate inputs without CUDA
execution. `chunk_fusion.cu` checks all scratch words after each fused phase,
not just final output. There are 48 synthetic extent/tail cases plus actual5
frozen panels, each with three phase comparisons, signed zeros, nonzero F32 state,
and guards. Independent HF gates remain separate from old/new equivalence.

Prepared root: `target/validation/gdn-chunk-fusion-20261007/prepared-v2`.
The first preparation failed only in the new test TU (namespace ambiguity);
the corrected fresh preparation compiles without warnings. No GPU test ran.

CPU checks: four Rust plan tests, two model chunk-plan tests, six fusion source/
extent/provenance tests, eight continuation preparation tests, five frozen math
CPU tests. Strict backend CUDA all-tests clippy passes with `-D warnings`.
Backend ignored tests are built, including actual owner/resource checks, one-row
passthrough and new 2/64/65/129-row resident launch-count checks with dirty scratch.
The two dedicated model count assertions derive from the backend plan; model
submission/metrics production code already uses that plan and was not changed.

PTXAS (not runtime timing): initial fusion16 registers, attention/delta/end-key
fusion16, output/state fusion20. All three have zero stack/spill bytes and no
shared memory. This is not an occupancy or latency claim. The unchanged matrix
and triangular kernels retain their arithmetic; full GPU acceptance is pending.

### Exact bounded GPU commands — only after ownership is granted

Run these serially from the repository root; all receipt paths must be fresh.
Each GPU child has a120-second bound and uses the existing authenticated
supervisor with CUDA library pins, drain and all-eight-idle checks. The commands
must not be executed by the current CPU-only owner.

```sh
. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
export NVIDIA_VISIBLE_DEVICES=all
python -B crates/ferrule-backend/tests/recurrent_bf16/chunk_gate.py run target/validation/gdn-chunk-fusion-20261007/prepared-v2/baseline --evidence target/validation/gdn-chunk-fusion-20261007/baseline-run
python -B crates/ferrule-backend/tests/recurrent_bf16/chunk_math_regression.py run target/validation/gdn-chunk-fusion-20261007/prepared-v2/continuation --evidence target/validation/gdn-chunk-fusion-20261007/continuation-run --execute-cuda
python -B crates/ferrule-backend/tests/recurrent_bf16/chunk_splitk.py run target/validation/gdn-chunk-fusion-20261007/prepared-v2/matrices --evidence target/validation/gdn-chunk-fusion-20261007/matrix-run
python -B target/validation/gdn-chunk-splitk-20261006/owned.py target/validation/gdn-chunk-fusion-20261007/prepared-v2/extra-jobs.json target/validation/gdn-chunk-fusion-20261007/fusion-stage-run
python -B target/validation/gdn-chunk-splitk-20261006/owned.py target/validation/gdn-chunk-fusion-20261007/rust-jobs.json target/validation/gdn-chunk-fusion-20261007/rust-run
```

Stop on any failed gate. The extra jobs execute 49 fused-panel cases and four
native observed replays, then compare all120 stages against retained independent
HF tensors (including production dense129 and Dk511/K512 cases). Original10
baseline, seven continuation/reset and14 matrix gates remain required. Source
and binary pins are checked before execution; reprepare if another owner changes
any pinned source. No timing flag is enabled. Matched unfused/fused timing for
actual5/dense64/dense129 remains follow-up after exact correctness, not a reuse
of the old scan/chunk timings.

