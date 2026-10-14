# W8A16 native performance and dispatch regressions

## Scope and arithmetic

`kernel_bench.cu` calls the production native entries directly, not a reimplemented
kernel. `dispatch.rs` exercises the real Rust owner/provider submission path.
Neither runs a model or a serving benchmark.

The optimized implementation retains:

```text
A_operand = BF16_RNE(A_f32), or the caller's BF16 input
W_operand = BF16_RNE(F32_RN(decode_E4M3FN(W) * original_dtype_scale))
D_f32 = F32 accumulation(A_operand * W_operand)
```

The vector path expands the **already BF16-rounded** values for F32 FMA. It does
not borrow the F32-profile kernel or fold/round scales early. Accumulation order
can differ, just as it can across Tensor Core tilings. The existing independent
F64 oracle tolerance (`1e-5 + 2e-6 * sum_abs_products`) is unchanged.

## Implementation and metadata

- M<=8: four warps/CTA; one output column per warp, specialized for one, two or
  four real activation rows. Four-row groups reuse weight decode. M=5..8 uses
  two groups. Packed weights require K divisible by four and an aligned base;
  otherwise a byte-load specialization handles tails. No shared memory.
- M>8: 16x32x32 WMMA tile, 256 cooperative loader threads, only two compute
  warps. Each loader decodes four neighboring bytes, using an aligned word load
  where legal. All 256 threads participate in barriers. The 3072-byte operand
  tile aliases the epilogue; the segmented balanced F32 reduction additionally
  uses 5/8/22 shared levels for K<=2048 / K<=16384 / larger K. Total shared
  bytes are 13312/19456/48128. The last three long-K levels use registers.
- Both paths: one launch, zero global workspace, no split-K or atomics, no
  repacked weight artifact, no additional dependencies, unchanged owner checks.
- Actual selected-kernel metadata is `W8A16FusedBf16Vector` for M<=8 and
  `W8A16FusedBf16Mma` otherwise. Both require the existing independent W8A16
  capability bit 14 / SM80+; F32/BF16 legacy bits do not provide a fallback.
  The family capability name is now `numeric-w8a16-fused-bf16` rather than
  claiming that every shape executes MMA.

## Bounded CUDA12.4 reproduction

From the repository root, use only the GPU owner's approved environment:

```sh
. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Do not run concurrently with another GPU owner. Never terminate foreign jobs.
Create a new evidence directory before building; do not overwrite a prior run.
The following example uses a pre-created `target/w8a16-microbench` directory:

```sh
nvcc -std=c++17 -O3 -arch=sm_86 -DFERRULE_CUDA_TARGET_SM=86 \
  -I crates/ferrule-backend/native/cuda --cudart shared --cudadevrt none \
  -L target/validation/qwen35-35b-current/toolchain/link \
  --ptxas-options=-v \
  crates/ferrule-backend/tests/numeric_w8a16/performance/kernel_bench.cu \
  -o target/w8a16-microbench/kernel-bench
ldd target/w8a16-microbench/kernel-bench
# Confirm libcudart.so.12 resolves to the sourced CUDA12.4 environment.
timeout 180s target/w8a16-microbench/kernel-bench 100
```

This standalone build intentionally needs no CUTLASS include directory: the
benchmarked W8A16 and F32-M1 native entries are original Ferrule kernels. Legacy
CUTLASS GEMM entries are unused. The Cargo regressions additionally compile and
exercise the normal production aggregate translation units with CUTLASS.

The matrix has M=1/2/4/8/9/32, five checkpoint-derived N×K shapes (512×2048,
2048×512, 8192×2048, 4096×2048, 2048×4096), two ragged shapes (131×129,
517×2051), both scale dtypes and both input dtypes: 168 W8A16 cases plus 14
separately labelled F32-M1 companion cases. Ragged shapes use scale origins
127/125; all cases use padded input/output strides. Host setup, uploads and the
independent CPU oracle run outside timing. The oracle checks every ragged
output, representative columns of each large shape, and all output canaries.

Each case captures 100 native calls, warms up three graph replays, then reports
median/min/max of seven CUDA-event replay durations divided by 100. The optional
argument is bounded to 1..1000 calls/graph. These are warm/repeated-artifact GPU
kernel measurements, not cold HBM, host dispatch, model accuracy or token/s.
Clocks are not locked; use repeated matrices and retain variability rather than
extrapolating a serving speedup.

Production-boundary tests:

```sh
timeout 1800s cargo test -p ferrule-backend --features cuda --release \
  --test numeric_w8a16 --test native_layout \
  -- --include-ignored --nocapture --test-threads=1
```

`dispatch.rs` covers packed/byte paths, scale boundaries inside a packed word,
partial four-row groups, the M8/M9 boundary, M16/M17 and M32/M33 tails, canaries,
F32/BF16 input parity and anti-F32-fallback/anti-premature-scale-rounding cases.
Run Compute Sanitizer memcheck and racecheck on the emitted `numeric_w8a16`
executable with `--ignored --nocapture --test-threads=1`, using `--error-exitcode 99`
and an external bound. Do not assume an artifact hash/path from another build.

## 2026-10-02 GPU-owner evidence

Full per-shape/scale/input timings, candidate results, source manifests, binary
hashes, linkage and sanitizer logs are in:

`target/validation/qwen35-35b-current/w8a16-kernel-opt/REPORT.md`.

RTX 3090 / SM86 / CUDA12.4.99, F32 input + BF16 scales, first baseline/final pair:

| N×K | M1 before → after (µs) | M32 before → after (µs) |
|---|---:|---:|
| 512×2048 | 298.527 → 15.340 | 321.628 → 81.654 |
| 2048×512 | 75.766 → 7.854 | 81.910 → 23.716 |
| 8192×2048 | 458.445 → 80.476 | 483.584 → 239.299 |
| 4096×2048 | 451.809 → 44.227 | 476.160 → 122.132 |
| 2048×4096 | 901.550 → 44.360 | 950.733 → 216.863 |
| 131×129, offset scales | 20.705 → 2.580 | 22.180 → 7.854 |
| 517×2051, offset scales | 298.383 → 23.583 | 323.604 → 91.904 |

M2/M4/M8 and all other dtype pairs are recorded, not omitted from the evidence.
The repeated full matrix improved all 140 W8A16 cases; the lowest observed
speedup was 1.82×. M1 still trails fused F32 on these shapes: this is an
optimization of genuine W8A16, not a claim of F32 parity or serving acceptance.

Rejected experiments are retained in the evidence: vector-only M32 regressed
8192×2048 (~0.47× baseline); register-only MMA and larger K128 shared tiles were
not consistently faster. Hardware BF16 conversion intrinsics were also tested
but the accepted implementation retains the original `bf16_rne` primitive.
Nsight Compute failed due to driver/tool incompatibility, so no hardware stall
counter claim is made. Source geometry and controlled kernel variants support
the under-parallelized operand-staging diagnosis.

Handoff: the model-owned assertion in
`crates/ferrule-model/tests/unit/transformer/numeric_precision.rs`,
`w8a16_model_precision_selects_fused_zero_scratch_backend_for_every_bucket`,
currently expects MMA for every M. Its owner must update the expectation to
vector for M<=8. It was inspected but not edited or executed by this backend
owner. No whole-model/serving run was performed.

## Segmented WMMA correction (2026-10-04)

The earlier tables describe the pre-fix implementation. The correctness fix
and its measured regression are documented in [../reduction/README.md](../reduction/README.md).
The updated benchmark includes M9 explicitly. Fixed bound and input generation
are unchanged. Evidence: `target/validation/w8a16-wmma-fix/performance.json`.
