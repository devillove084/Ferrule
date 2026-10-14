# BF16 gated RMSNorm exact regression

## Fix and scope

The frozen Capital decode0 layer0 norm followed exact projections, GDN output,
and F32 state. Its old sequential square sum caused two BF16 output differences
at1908/1910. The production norm now mirrors pinned Torch contiguous F32 mean
geometry for widths1..4096. Width128 uses four contiguous values per lane and
a32-lane halving tree. Other widths retain the reference four-accumulator,
vector-alignment head/tail order. Alignment refers to the virtual materialized
F32 square tensor, not the BF16 input address. Mean uses a rounded F32 reciprocal
multiply, including non-power-of-two widths.

Explicit RN arithmetic, BF16 normalized and weighted boundaries, F32 SiLU, and
final BF16 output remain intact. Inactive rows participate in all barriers.
Widths above4096 retain the unchanged admitted scalar path; HF parity is not
claimed for that wider geometry. One owner-stream kernel remains, with no added
allocation, workspace, copies, or host synchronization; static shared memory
is2048 bytes. ABI, owner guards, F32 SSM, conv, recurrence, profiles, and other
norm types are unchanged. No chunk fusion or model edits.

## Dedicated test

The gated_norm.py prepare command verifies retained Capital/Hello captures,
checkpoint weight payloads, production sources, and the archived old kernel.
It generates a test-only instrumented production tree. Observer output must
match uninstrumented output. A separate authenticated HF process observes the
unmodified pinned producer; original captured real outputs remain exact guards.
The gated_norm.cu harness checks both-end canaries and input immutability.

From the repository root, source
 target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
then prepare with GPUs hidden:

    CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=void LD_LIBRARY_PATH= /opt/conda310/bin/python -B crates/ferrule-backend/tests/recurrent_bf16/gated_norm.py prepare target/validation/gdn-gated-norm-rerun-prepared

With exclusive GPU ownership and CUDA_VISIBLE_DEVICES=0:

    python -B crates/ferrule-backend/tests/recurrent_bf16/gated_norm.py run target/validation/gdn-gated-norm-rerun-prepared --evidence target/validation/gdn-gated-norm-rerun-receipts

Both output directories must be fresh. Each GPU process is bounded120seconds,
with authenticated subreaper custody, pinned CUDA libraries and all-eight-GPU
idle/drain checks. Native runtime is CUDA12.4/sm86. The independent oracle is
Torch2.13cu129/Transformers5.5.3 in a separate process, not a Ferrule CUDA12.9
runtime dependency.

## Results (2026-10-07)

Evidence: target/validation/gdn-gated-norm-20261007/FINAL.json.

-137 cases, zero failures: five actual Capital/Hello cases and33 synthetic
 widths crossed with rows1/3/17/32, including width128 and alignment tails.
-133 tree cases: exact variance, inverse, normalized, weighted, SiLU, precast,
 and output. Four width4097 cases: exact old/new fallback stability only.
 Capital old output still differs2/4096; current all stages are exact.
-88 selected recurrence cases:21,342,720 state/output words exact.
-107 stability cases, including10 below threshold:2,914,300 words exact.
-Actual Hello:4096 output and524288 F32 state words exact; output2603=0x3911.
-Four authenticated Rust GPU gates pass: actual cached replay, frozen reduction,
 prefill/decode/segmented continuation including resource counters, and all
 operand owner/extent rejection. Four CPU tests pass. Strict backend CUDA
 all-tests clippy passes with GPUs hidden and -D warnings.

Actual Capital norm CUDA events:20 warmups each,30 alternating AB/BA rounds,
eight individually timed calls/sample; complete checks outside timing.
Old/new median35.968/6.400us; p95 36.1664/6.5984us. Registers28/27; both zero
stack/local/spills; new shared2048 bytes. These are measured norm-only latencies,
not model throughput; old output has the known two-word defect.

No oracle/tolerance changes. Mixed/full-model validation deferred while another
agent edits provider safety. Final evidence includes source pins and GPU release.
