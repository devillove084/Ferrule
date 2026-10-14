# Resident BF16 router GemmEx

`CudaBf16GemmEx::execute_bf16_view` uses a shape warmed in the existing prepared
pool. Its packed BF16 input, resident BF16 weight and BF16 output are passed
straight to the handle's BF16 A/B/C, F32-compute GemmEx call. It performs no FP8
decode, input/weight staging copy, allocation, stream/workspace rebind or handle
creation during invocation. There is one submission instead of pack+decode+GEMM.

The existing prepared pool admission remains conservative: full maximum staging,
32MiB explicit cuBLAS workspace and16MiB external reserve, charged once per owner.
Adding router shapes does not grow the actual model's QKV-sized maximum allocation.
No second pool is created. `CUBLAS_WORKSPACE_CONFIG=:0:0` must be established by
the launcher; CUDA12.4/sm86/cuBLAS120405 only.

The contract is packed contiguous buffers matching the exact plan, alignment256,
M1..32, N/K multiples128, owner/context identity, disjoint output/read operands,
and no overlap with provider storage. Strided views and arbitrary unretained
capture are rejected before submission. This eager API follows normal caller
last-consumer ownership, not a new graph API. The model retains original resident
weight artifacts and input/output buffers through its existing work holds; Unknown
retains them with the provider and blocks reuse.

Model dispatch selects only native-BF16 `RouterLogits` in the existing Mixed
profile. Numeric FP8 router selection, other profiles, output head and A/B remain
unchanged. N32 A/B native probes are diagnostics, not admitted new geometry.

## Bounded evidence

`prepare.py` extracts existing full-image actual producer inputs and original
resident BF16 weights; it checks exact HF input identity. No whole-model capture.
Original weight payload hashes are separately audited against the independent
HF capture provenance. Four router cases: Hello layers0/3 M1, Capital layers0/1 M5.

- Direct probe and production model dispatcher:0/3072 router words differ.
- Legacy projection: Hello3 differs1/256 (index69, max0.03125); other three exact.
- Selected IDs unchanged in all96 slots. The router kernel itself is unchanged.
- Remaining route-weight word differences:3/8,1/8,26/40,18/40 (48/96 total).

`route_evidence.py` observes full-expert F32 softmax, selected probabilities, their
sum and normalization using pinned native HF. It also evaluates selected-only
exponentials with serial and native reduction. Existing actual weights match the
selected-only serial algorithm exactly; replacing only that sum tree does not
reproduce full-softmax-then-selected-renormalization. Algebraic equivalence is not
bit equivalence. A later router arithmetic change needs its own contract/evidence;
this patch neither changes tie policy nor claims route-weight exactness.

GPU tests cover frozen real cases, synthetic M1/M5, stride/short-span/alias/foreign
owner/capture rejection, reuse/reclaim, model role dispatch, and retained Unknown
custody. The original Unknown test was corrected to inject before a nested router
call's successful completion proof; the failed receipt is retained.

Exact source/binary/library/fixture pins, authenticated commands and results:
`target/validation/bf16-resident-router-20261007/`. No full40 run in this cycle.
