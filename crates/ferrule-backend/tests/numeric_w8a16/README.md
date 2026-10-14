# Numeric W8A16 backend handoff (updated 2026-10-02)

## Independent scalar CUDA triangle (2026-10-02)

`reference.cu` is a separate test-only translation unit enabled by
`FERRULE_CUDA_TEST_ORACLE=1` together with the `cuda` feature. It includes only
CUDA runtime and standard headers, uses arithmetic E4M3FN decode, residue/tie
BF16 RNE, absolute checkpoint block coordinates, and one serial F32 RN-FMA dot
per output. It does not include production helpers or reproduce the production
warp/MMA reduction. Normal builds neither compile nor link it.

`scalar.rs` compares **each** CUDA result with the independent host F64 dot,
using exactly the existing `gpu.rs` bound `1e-5 + 2e-6 * sum(abs(a*w))` after
operand rounding. This is the established synthetic operator criterion, not a
universal error theorem or the full40 logit criterion. CUDA-to-CUDA and scalar
CUDA-to-host serial F32 bit differences are diagnostic, not acceptance gates.
The test also checks both production input APIs, output canaries and unchanged
input. It writes a create-new-only JSON report when
`FERRULE_W8A16_TRIANGLE_REPORT` is set.

Reproduction (one GPU owner, GPUs idle first; other builds hide devices):

```sh
. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
export CARGO_TARGET_DIR=target/validation/w8a16-scalar-reference-20261002/cargo
export FERRULE_CUDA_TEST_ORACLE=1
CUDA_VISIBLE_DEVICES='' cargo test --offline --locked -p ferrule-backend --features cuda --test numeric_w8a16 --no-run
# Verify the built executable loads libcudart.so.12 before GPU execution.
# Choose a NEW report filename; the test deliberately never overwrites evidence.
FERRULE_W8A16_TRIANGLE_REPORT=target/validation/w8a16-scalar-reference-20261002/operator-triangle-next.json cargo test --offline --locked -p ferrule-backend --features cuda --test numeric_w8a16 scalar::independent_scalar_cuda_host_production_triangle -- --exact --ignored --test-threads=1 --nocapture
```

Measured on SM86/CUDA12.4: 320 cases, 92,496 outputs, zero violations; full
`numeric_w8a16 --include-ignored` suite 15/15 passed. M=1/2/4/8/32, packed and
ragged K, four aligned/crossing origin pairs, BF16/F32 scales, finite E4M3FN
codes and signed RNE midpoint inputs are covered. This supersedes the historical
unexecuted backend-test note below **only for these tests and this run**.

Evidence: `target/validation/w8a16-scalar-reference-20261002/`. The archived
report records scoped operator math PASS, full40 implementation-equivalence
FAIL (1,837,493 failures), and task quality NOT RUN. Its report-composition
workflow is retired; the evidence and original thresholds remain unchanged.
The scalar reference is not a full-model oracle; diagnostic reduction replay
is not model-quality acceptance.

## Marlin comparison boundaries

The local `target/validation/marlin-audit/REPORT.md` compares installed vLLM
`0.28.0+cu129`, commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`, not upstream
main. Ferrule is an independent implementation, not a copied Marlin kernel or
repack format. Any future source port must preserve applicable Apache-2.0
license and attribution. Installed source identity is not binary reproducibility.

The recorded SM86 comparison covers only two real shapes: M1 shared gate and
M5 QKV. Checkpoint weights/scales were byte-verified and activation BF16 words
matched. For these BF16-scale cases, 17,825,792 decoded weight operands matched
at the BF16 boundary. This does not generalize to arbitrary F32 scales,
overflow or extreme values. This Marlin path decodes and multiplies scales
before MMA, not after grouped accumulation.

- **Output:** Marlin returns BF16 for BF16 input; this Ferrule operator returns
  F32. Raw F32 differences are not a like-for-like correctness verdict. Even
  after rounding Ferrule output to BF16, the two cases had 0 and 8 differing
  words respectively; equal operands do not imply identical reductions.
- **Gate/SiLU:** Marlin rounds gate/up GEMM outputs, SiLU intermediates and
  activation output at BF16 boundaries. The compared Ferrule path keeps
  gate/up and SwiGLU in F32, rounding inputs at the next projection.
- **Combine:** Marlin rounds down output and route weights to BF16 before
  multiplication; the compared Ferrule path uses F32 multiply/add. Restoring
  top-k slots or scheduling by expert does not establish expert-ID combine order.

The bound `1e-5 + 2e-6 * sum(abs(A*W))` tests this operator's F32 output. It must
not be applied to BF16-rounded output to declare Marlin wrong, nor relaxed or
paired with a changed output dtype to hide a mismatch. These scoped operator
observations certify neither whole-model quality nor equivalent vLLM performance.

## Explicit API; no model/runtime/default-profile change

Import from `ferrule_backend::cuda::operators::linear`:

```rust,ignore
let plan = NumericFp8LinearPlan::new(
    layout, rows, 0, NumericFp8Precision::W8A16Bf16F32Accumulate,
)?;
let mut workspace = ops.numeric_fp8_linear_workspace(plan)?; // outside capture
// Existing immutable numeric payload upload; no repack/requantization.
let artifact = ops.upload_validated_numeric_fp8_linear(&payload)?;
ops.numeric_fp8_linear_into(
    &artifact, &f32_input, &mut f32_output, &mut workspace,
    plan, activation_stride, output_stride,
)?;
// Alternative, not an additional launch: already BF16-rounded activation.
ops.numeric_fp8_bf16_linear_into(
    &artifact, &bf16_input, &mut f32_output, &mut workspace,
    plan, activation_stride, output_stride,
)?;
```

The F32 API rounds A to BF16 **inside the fused kernel**, not via a hidden
allocation or conversion launch. The BF16 API takes `CudaBf16Buffer` and rejects
all other precision profiles. Strides are input/output elements, not bytes.
Output is F32; model code decides whether and where to add BF16 output rounding.
Cache/reuse/graph keys must include precision, selected kernel, input dtype,
strides, shapes and exact resident operands. Do not relabel a captured F32-input
node as BF16-input. KeepAll and bounded residency logic are not changed here.

Model owner must add an explicit opt-in mapping; none is installed by this work.
`F32Tf32x3` remains fused IEEE F32 M1 / tiled TF32x3 multirow, with unchanged
native kernels and tolerances. `Bf16RneF32Accumulate` remains the existing separate
pack/decode/GEMM path, never advertised as fused.

## Real implementation and supported shapes

`native/cuda/cutlass/numeric_w8a16.cuh` is an original Ferrule kernel compiled
in the existing core TU and advertised through the existing CUTLASS provider.
For M<=8, four independent warps/CTA each own one N column and up to four
real M rows. Packed four-byte weight loads (or guarded byte loads for odd K /
unaligned bases) feed explicit F32 decode-times-scale, BF16 RNE operands,
F32 FMA and warp reduction. The BF16 boundary is identical to the WMMA path;
this is **not** the F32 profile under another name. M=1/2 use specialized row
counts; M=3..8 use guarded groups of four, sharing weight decode across rows.

For M>8, eight warps/CTA cooperatively load/decode a 16x32x32 operand tile;
two of them execute BF16 WMMA with F32 accumulators. Packed four-byte loads
reduce weight instructions. Scales are still selected per element, including
unaligned scale origins crossing inside a packed word. All eight warps join
both block barriers. The epilogue reuses operand storage and writes only valid
strided outputs. WMMA resets its F32 accumulator every 128 K elements and
merges those partials with an explicit balanced F32-RNE carry tree. See
[reduction/README.md](reduction/README.md) for the fixed real-input failure,
seeded regression, resource bounds and measured performance cost. Neither
path materializes decoded weights globally.

Compiled capability: `CutlassKernelId::NumericFp8W8A16Fused` (bit/id 14),
family name `numeric-w8a16-fused-bf16`. The capability bit, SM80+ requirement,
precision profile and ABI remain unchanged. Actual selected-kernel metadata:

- M<=8: `W8A16FusedBf16Vector`, `numeric-w8a16-fused-bf16-vector`.
- M>8: `W8A16FusedBf16Mma`, `numeric-w8a16-fused-bf16-mma`.

See [performance/README.md](performance/README.md) for bounded CUDA12.4
microbenchmark reproduction, measured improvements and rejected variants.
Compile target SM80+, verified compile on SM86 only. This is not native FP8 MMA.
No CUTLASS GEMM invocation, global dequant buffer, split-K/atomic reduction,
CPU fallback or full floating-point weight materialization in this path.

M/N/K are positive, with existing upper guards M <= INT_MAX-64,
N <= INT_MAX-128, K <= INT_MAX-32 and checked isize-representable extents.
Ragged M/N/K, arbitrary nonnegative 128-block origins and padded strides work
by zero-filling operand tails and masked epilogue. Flat 64-bit tile indexing
uses a capped grid-stride loop; no grid-Y dimension limit. F32 input/output
require 4-byte alignment, BF16 input 2 bytes, scales their element alignment.

BF16 conversion may overflow/underflow. Scale validation still rejects NaN,
nonpositive/nonfinite scales, NaN E4M3FN and nonfinite *F32* dequantization;
it does not promise finite BF16 conversion or accumulation. TensorOp subnormal
and summation behavior is not a serial F32/bitwise IEEE contract. Independent
F64 and serial F32 oracles use the explicit BF16 operand boundary. No original
F32 tolerance was changed.

## Accounting and custody

- `kernel_launches() == 1` on successful submission; owner counter records it.
  The launch count is structural, not a profiler measurement in this handoff.
- `workspace_requirements().bytes`, `activation_bytes()`, `weight_tile_bytes()`,
  `tile_rows()` and `tile_count()` are zero: these describe **global** storage.
- `workspace_allocation_grant_bytes() == 0`. Zero allocator requests retain
  owner identity but acquire no device block. Workspace construction still
  belongs outside capture, even for zero bytes.
- `fused_shared_bytes_per_cta()` is `Some(0)` for the vector path and
  `Some(13312)` for WMMA K<=2048, `Some(19456)` for K<=16384, and
  `Some(48128)` above that. This includes the 3072-byte operand/epilogue tile
  and bounded reduction tree, **not** HBM residency or allocator reserve.
  CUDA12.4/SM86 ptxas reports zero stack/spills for all revised specializations;
  other compilers/architectures still require resource validation.
- Encoded logical storage remains N*K + scales. Physical grant is
  `grant(N*K) + grant(scale_bytes)` for two distinct allocations, available as
  `artifact_allocation_grant_bytes()`. Do not round their sum as one allocation.
- Caller input/output storage and any optional upstream BF16 conversion are
  separate budget classes. For example a BF16 staging request M*lda*2 has its
  own allocator grant; this kernel never silently creates it. Replacement
  peaks include old+new grants until retirement; shared segment slack, graph
  objects, driver code/module memory and quarantine are separate as before.
- Retained compatibility scratch must remain within declared logical budget,
  correctly owned/aligned/disjoint even when unused. Preflight failures do not
  poison; native submission failures poison and cannot be reset. Submission is
  not completion. Record an owner event after launch, wait before cross-stream
  use and retain all operands/workspace/graph resources through completion.
- No new packages, CUDA libraries, downloads or vendored source; dependency
  growth budget and actual dependency growth are zero. No host-KV/common
  semantics, model or runtime files changed by this implementation.

## Read-only vLLM provenance and non-equivalence

Inspected the locally installed `target/vendor/vllm028-py312/lib/python3.12/site-packages/`:
`vllm-0.28.0+cu129.dist-info/METADATA` declares Apache-2.0. No import/server/GPU
initialization was used for inspection; no source was copied from vLLM.

- `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py`:
  SHA256 `72c324a9de072e60c1f126ad3d5e7f077f6a8752590435e088634f798683d12a`.
  `apply_fp8_marlin_linear` calls `marlin_gemm` for float8_e4m3fn W8A16;
  `prepare_fp8_layer_for_marlin` repacks weights, converts scales to original
  dtype, expands block scales to groups, and fuses exponent bias into scales.
- `vllm/model_executor/kernels/linear/scaled_mm/marlin.py`:
  SHA256 `eb4f0c3494d320ed71cc06ecff40eafa6695bbe60d10c178d67df0484bdf0b8f`.
  The SM86 weight-only route is explicitly supported; W8A8 is not this profile.
- `marlin_utils.py` has `USE_FP32_REDUCE_DEFAULT = True`.
- Existing CUTLASS checkout `e05f953a5b3d38adc240df2ff928e0421c2abba3`,
  `LICENSE.txt` BSD-3-Clause, is reused unchanged by the provider build.

This implementation uses original scale dtype, no exponent-bias transformation,
no Marlin repack/scheduler and F32 output. It is W8A16-class arithmetic, **not**
Marlin bitwise equivalence or a matched end-to-end profile. The archived vLLM
TP4/DP2/EP8 versus Ferrule TP1/EP8 difference in `docs/PERFORMANCE_EVALUATION.md`
still applies. No throughput, quality, or approaching-vLLM claim is made.

## Validation and owner-run gates

- CPU-only: `cargo test -p ferrule-backend --test numeric_w8a16
  --test cuda_refactoring_contracts --test cuda_scratch_metadata`: 4 + 21 + 3 pass.
- CUDA-linked CPU-only new suite: 7 pass, 3 GPU tests ignored, with
  `CUDA_VISIBLE_DEVICES=''`. Includes native preflight on fake non-dereferenced
  addresses and independent capability/physical-plan tests.
- Existing F32 `explicit_profile_identity_and_exact_budgets` and
  `m1_dispatch_zero_budget_and_shape_boundaries`, plus BF16
  `exact_budget_and_invalid_metadata`: 3 passes with devices hidden;
  their 10 GPU tests remain ignored.
- CUDA 12.9.86, GCC toolset12, SM86: backend `cargo check --features cuda --tests`
  and `cargo test --features cuda --test numeric_w8a16 --test cutlass_numeric_fp8_f32
  --test cutlass_numeric_fp8 --lib --no-run` passed. Optimized standalone native
  compile also passed. Use installed cu129 cuBLAS headers/lib; the JIT toolkit
  alone lacks cuBLAS headers. No global toolkit/symlink changes.
- Strict Clippy passed for CUDA backend lib, new test target and the related
  source-contract target (`-D warnings`, no lint suppression flags). Targeted
  rustfmt checks and backend `git diff --check` passed.
- Dedicated ignored tests cover M1/2/4/8/32, K1/3/35/129/259/8193, ragged N,
  BF16/F32 scales, non-block-aligned origins, BF16 and F32 input, output canaries,
  owner/profile/alias/budget guards, graph replay, cross-stream event reuse and
  no request-time allocation/residency growth. An additional ignored unit test
  in `poison.rs` verifies injected quarantine on both input APIs and capture.
  Injection is not a claim to reproduce hardware asynchronous failure.

**GPU tests have NOT run.** Unified GPU owner must run these plus memcheck/
racecheck and the explicit new-profile full-model oracle before enabling any
serving profile. GPU graph execution, BF16 subnormals, quality, latency and
end-to-end throughput remain unvalidated. Profile tuning (larger tiles,
pipelining, small-M scheduling) requires measurement; this first implementation
is not the Marlin optimized schedule. No commits were created.
