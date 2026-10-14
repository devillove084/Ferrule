# Resident scalar and positive-N BF16 GemmEx

`Bf16GemmExPlan::new_resident` admits M1..32, positive native i32 N/K and checked
storage arithmetic. Packed lda=K/ldb=K/ldc=N are not padded. Resident operand
pointers require BF16 element (two-byte) alignment, not tensor-core preferred
alignment. Numeric FP8 retains positive128-wide blocks, origins0/BF16 scales and
its previous alignment checks. Resident identity still cannot enter FP8 decode.

Mixed adds only semantic `SharedExpertOutputGate` to the existing resident roles.
Output head and other precision profiles are not changed. N1 shapes warm before
invocation on the same image-owned handle/storage/workspace. Workspace remains
once-bound and256-byte aligned; no per-projection reserve, decode, copy, alloc,
fence or rebind. The full-header maximum remains67,272,704B logical storage,
69,206,016B physical grant plus16,777,216B external reserve. Total6,846,743,296B
under7GiB, with669,449,472B headroom including the existing512MiB margin.

## Evidence

`target/validation/bf16-resident-scalar-20261007/`:

-4 original independent HF scalar cases (M1 and M5),12 words, all exact in backend
  and actual Mixed model dispatch. Fresh QA layer15's -2.046875 legacy result is
  corrected to original HF -2.0625.
-40 deterministic positive-tail cases: M1/M5, N1/3/7/8/31, K1/7/31/2048, pointers
  offset by2 bytes. Compared to an independent public CUDA12.4/sm86 BF16-output
  `cublasGemmEx` process using the unchanged compute/math/atomics/workspace settings.
  All44 cases total1212 output words: zero differences. Input and weight GPU
  roundtrips are bit-exact. Physical-budget, invalid dimensions, alias, short
  extent, foreign owner, packed-stride, unretained capture checks pass. Pool
  reclaims after completion; invocation counters prove one launch/no extra work.
-Mixed tiny and one original full40 regression pass. No full40 references changed.
-Original fresh oracle manifest102c798baeb5200bc8ed8386eabed6146870b3b6da2a90fc6010677041352c7e
  reused directly; no regeneration. QA all9 predictions passes, arithmetic first6
  predictions passes, then arithmetic step6 has194352/248320 failing logits,
  max.59375,19 route ID differences, despite exact next-token choice.
-Free-greedy/lifecycle/cold progression waits until ALL teacher-forced cases pass.
  Chinese was not reached. This is not a completed fresh-quality acceptance.
-One targeted arithmetic replay uses the same current full image, no fork,
  layers25..28 only. Metrics reproduce the original failed step exactly. Layers25
  and26 are exact. Layer27 projections, QK norms, RoPE and KV are exact; first
  observed discrepancy is gated-attention output196/4096 words, all query head6,
  max.00152587890625. Native HF layer replay validates5/5 original outputs/KV.
  Output projection on actual input is exact. `frozen-attention-boundary/` stores
  exact Q/K/V/gate, independent ungated/gated expected, and actual gated output.
  Raw attention versus sigmoid/multiply remains unresolved; no speculative fix.

Concurrent serving-agent files temporarily broke composition clippy. They were
not edited here. Current binaries were rebuilt after those writes stabilized;
source snapshots and the concurrent-write scope are recorded. No serving CLI
changes belong to this task. The scalar/unit tests and current strict clippy are
separate from a throughput or universally bit-exact GemmEx claim.
