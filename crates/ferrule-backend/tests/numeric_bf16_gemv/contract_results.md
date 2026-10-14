# Frozen routed-M1 call-contract result

This supersedes the execution status (not the immutable preparation evidence) in
`contract.md`. All four prepared plans completed under authenticated supervision.
The existing production backend, provider ABI and model dispatch are unchanged.
No full40, installations, toolkit changes, oracle rewrites or tolerance changes.

Evidence: `target/validation/capital-gemv-contract-20261006/`:

- `GPU-FINAL.json`: conclusion, exact call contract and integration requirements.
- `GPU-MATRIX.json`: binary-audited full matrix, bit positions and output hashes.
- `hf-calls.jsonl`: 136 BF16 GemmEx calls plus eight F32 diagnostic calls.
- `*-supervisor.json`: four fresh authenticated execution/drain receipts.
- `GPU-CPU-CHECKS.json`: nine Python tests and one existing host-policy test.
- `GPU-RELEASE.json`: authenticated drains and three all-eight-GPU idle checks.

## Actual native BF16 contract

Pinned Torch `2.13.0+cu129`, revision
`cf30153c4c131c8164ee7798e5022d810682e2cb`. Plain and interposed HF are byte-exact
to the original BF16 output on all eight fixtures, with one output hash per case
across 16 repetitions. Both profile the same `internal::gemvx::kernel`, BF16
input/output, F32 accumulator, configuration7 (full name in `GPU-MATRIX.json`).
The public runtime API is conclusively **GemmEx**, not inferred from this name:

```cpp
// Private handle: bind stream before workspace. No subsequent stream reset.
cublasSetStream(handle, owner_stream);
cublasSetWorkspace(handle, caller_owned_workspace, 32 << 20);
cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED);
// alpha/beta are float 1/0, and all operand addresses are 256-byte aligned.
cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, 1, K,
             &alpha, weight, CUDA_R_16BF, K, input, CUDA_R_16BF, K,
             &beta, output, CUDA_R_16BF, N,
             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

Down: N2048/K512. Gate-up: **full concatenated N1024/K2048**, not sliced N512.
No F32-output conversion is substituted for the actual BF16 oracle.

## Numeric matrix

Each entry is unequal BF16 words against the original output, maximum over16
repetitions. The entire matrix and every repeated binary output are identical
between cuBLAS120405/CUDA12.4 and cuBLAS120901/CUDA12.9 on sm86.

| Fixture | Matched GemmEx | No-workspace GemmEx | Default-math GemmEx | Atomics-allowed GemmEx | Matched TST | Prior TST | TSS diagnostic | F32 GemmEx diagnostic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 down | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 57 down | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 146 down | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| 151 gate-up | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 200 down | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| 226 down | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 231 gate-up | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 238 down | 0 | 3 | 0 | 0 | 0 | 3 | 0 | 0 |

All native configurations are stable across16 repeats, including alternating
output reset patterns with beta0. Input immutability and allocation guards pass.
Matched BF16 GemmEx is exact on14,336 words per pass,229,376 per library across16
repeats. TSS/F32 entries are diagnostic comparisons only, not alternate oracles.

The zero-workspace/prior-TST differences (actual vs HF, zero-based indices):

| Fixture | Index | Actual bits | HF bits | Absolute difference |
|---|---:|---|---|---:|
| 146 down | 1620 | `32e2` | `32e1` | 1.1641532182693481e-10 |
| 200 down | 1855 | `37dc` | `37dd` | 1.1920928955078125e-7 |
| 238 down | 1155 | `34e0` | `34e1` | 1.862645149230957e-9 |
| 238 down | 1278 | `b3d4` | `b3d2` | 9.313225746154785e-10 |
| 238 down | 1825 | `b76d` | `b76e` | 5.960464477539063e-8 |

**Workspace-sensitive arithmetic dispatch is isolated:** removing workspace alone
from matched GemmEx restores the five differences on both libraries. Matching
workspace also makes TST exact. Neither a packing defect nor a12.9-vs12.4
incompatibility is supported for these fixtures. No nondeterminism was observed;
this does not prove the proprietary kernel has no split-K/internal atomics or
establish its exact reduction tree. Math16 alone does not prohibit split-K.

## Why no unsafe in-place backend patch

The executable arithmetic candidate is already `contract_probe.cu`'s
`gemmex-matched` configuration. Production admission is a separate resource
change, not a safe one-line replacement of the existing GEMV call:

- Existing `Bf16M1Gemv` admits only M1/N2048/K4096,tile257, TSS/F32 output,
  zero workspace, and a16MiB external-provider reserve. Preserve that profile.
- `NumericBf16Gemv` owns a handle and stream, **not workspace storage**.
  `prepare_numeric_fp8_provider` retires initialization storage before returning.
  Binding that temporary storage as persistent cuBLAS workspace would dangle.
- New32MiB workspace must have explicit caller admission and a retained owner
  lease. It is additional to decoded operands, BF16 output staging and measured
  provider growth; it cannot be hidden inside the existing16MiB reserve.
- Use native BF16 output and full-width geometry, then losslessly widen if the
  operator interface requires F32. Do not round F32 output as a replacement.
- New profile needs stream/context checks, no allocation during launch/capture,
  warmup/growth admission, graph replay lifetime, and unknown-completion
  quarantine retaining **both** handle and workspace. Success-path probe cleanup
  is not evidence that production error/capture lifecycle is safe.

Next bounded integration gate: a separate explicit backend profile with the
above resources, all114 existing frozen projection cases, CPU shape/span/alias/
budget/overflow rejection and GPU lifecycle/capture/failure tests. Keep legacy
profile/defaults and model dispatch unchanged until those gates pass. No new
expert-ID, per-index or tolerance rule is needed. No full40 is needed for this
integration gate.

## Pins and scope limits

CUDA12.4 compiled the same host-only probe used in both isolated native runs;
CUDA12.9 was diagnostic only. Recorded runtime versions, ldd, source/binary/
fixture/provider pins and custody checks passed. The initial prepared pin set
omitted CUDA12.4 `libcublasLt.so.12`: its resolved path is in ldd and its hash is
recorded **post-run** in `GPU-FINAL.json`, not misrepresented as a pre-run pin.
There was no install or library change. A future fresh plan should include it.

Existing `PREPARED.json`, `FINAL.json` and all pinned preparation files remain
unchanged. The new CPU-only audit can be repeated with a fresh output filename:

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none /opt/conda310/bin/python -B -m unittest discover -s crates/ferrule-backend/tests/numeric_bf16_gemv -p 'test_contract*.py' -v
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none cargo test -p ferrule-backend --no-default-features --test numeric_bf16_gemv
```

GPU ownership released: four authenticated process trees proven drained, then
three fresh observations of all eight devices at0MiB/0% with no compute process.
