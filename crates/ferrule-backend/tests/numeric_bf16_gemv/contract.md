# Frozen routed-M1 cuBLAS call-contract gate

## CPU-only result

No GPU/device query, model dispatch change, native backend patch, full40, oracle
rewrite, or tolerance change was made. The chunk-continuation owner keeps GPU
ownership. Only new dedicated GEMV test/probe files were added.

Prepared evidence: `target/validation/capital-gemv-contract-20261006/`.
The actual frozen BF16 HF output remains the sole oracle. The17-word mismatch
between BF16 `F.linear` and rounded F32 output at M5 attention-output is retained
as prior evidence, not generalized away or used to replace expectations.

## Concrete public call-contract difference

Pinned installed Torch:2.13.0+cu129, git revision
`cf30153c4c131c8164ee7798e5022d810682e2cb`.

The installed binary's BF16 cuBLAS helper calls **`cublasGemmEx`**, not the public
`cublasTST/TSSgemvStridedBatched` functions used by the earlier CUDA12.4 probes.
`torch-bf16-helper.asm` records the bounded disassembly: BF16 A/B/C types14,
compute type68 (F32 unless pedantic requested), algorithm99
(`CUBLAS_GEMM_DEFAULT_TENSOR_OP`), and temporary math-mode setup/reset.
`CuBLASReductionOption::DisallowReducedPrecisionAllowSplitK` is value1;
the helper maps it to math flag16
(`CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION`). It is **not** an instruction
to ban split-K. A `gemvx` kernel name does not identify its public caller.

The installed `cuBlasCommonArgs.h` explains the row-major transpose convention.
For frozen `[1,K] @ [N,K].T`, expected cuBLAS arguments are:

```
cublasGemmEx(handle, T, N, N, 1, K,
             &float_one, W, CUDA_R_16BF, K,
             x, CUDA_R_16BF, K,
             &float_zero, y, CUDA_R_16BF, N,
             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
```

Thus down is public GEMM `(m,n,k)=(2048,1,512)` with leading dimensions512/512/2048.
For the two up blockers, the original HF executes **concatenated gate-up**:
`(m,n,k)=(1024,1,2048)`, not isolated N512. The test reconstructs the original
width only by concatenating the two already pinned decoded-weight/output halves;
inputs must match. There is no expectation synthesis.

The pinned-revision `CublasHandlePool.cpp` download succeeded. It installs the
current stream and explicit workspace, interprets `CUBLAS_WORKSPACE_CONFIG`
`:4096:8` as32MiB, and resets CUDA math mode to default before the GEMM helper's
own reduction flag. Its explicit deterministic atomics switch is ROCm-only.
The new runtime observer must confirm CUDA's actual atomics mode rather than
infer atomic use/non-use from determinism settings.

The other requested source downloads failed/interrupted, so those source bodies
are not claimed as inspected. Installed binary/header evidence supplies the
contract above. Public runtime arguments, exact alignments, actual workspace
and atomics modes are **still to be confirmed**, not fabricated from the
kernel name. All library/source files inspected are hash-pinned in PREPARED.

Retained native kernel identity is `internal::gemvx::kernel` with BF16 input,
BF16 output, F32 accumulator, configuration7, and strided-batched tensor
arguments; the full identity remains in the original supplemental manifest.
It alone does not prove split-K, atomic accumulation, or nondeterminism.

## Existing repeat evidence

Two independently supervised CUDA12.4 operand probes reproduce the same results:
full-width TST/TSS misses expert146_down1 word, expert238_down3 words, and
expert200_down1 word. Tile257 changes those counts to1/1/0. These are repeated
counts, **not a controlled fixed-buffer repeatability proof**. All decoded
operands were previously verified identical; no packing-bug claim is supported.

The frozen gate includes the prior seven blockers plus expert200_down:
19down,57down,146down,151up,200down,226down,231up,238down. Expert IDs select
regression payloads only; no arithmetic/backend rule branches on them.

## Prepared bounded execution

Four explicit90-second plans in `PREPARED.json`, each under the existing
authenticated subreaper with immutable source/binary/library/fixture pins:

1. `hf-plain`:16 native BF16 `F.linear` repetitions/case on fixed operands,
   exact BF16 expectations, output hashes, layout/alignment, and one kernel
   profile/case. F32-output conversion is separately labelled diagnostic.
2. `hf-logged`:same run with a pinned pass-through `cublasGemmEx`/workspace
   observer. A CPU-only mock verifies forwarding and result preservation.
   Its BF16 outputs must equal the plain run. The logger records dimensions,
   LDs/types, compute/algo, math/atomics/pointer mode, scalars, alignment,
   workspace and status. It does not change the call or returned output.
3. `native-129`:CUDA12.4-built host-only probe with separately pinned native
   cuBLAS12.9.1/runtime12.9 libraries. No Ferrule device kernels. Expected runtime
   cuBLAS version120901 must be reported by the loaded provider.
4. `native-124`:same binary and operands with cuBLAS12.4.5/runtime12.4,
   version120405. Separate process prevents mixing provider versions.

Both native plans are blocked unless observed HF call metadata matches the
prepared contract and the logger preserves native outputs. Eight fixed
configurations,16 repetitions/case: matched BF16 GemmEx; no-workspace GemmEx;
default-math GemmEx; atomics-allowed GemmEx; matched TST; prior-settings TST;
matched TSS/F32 diagnostic; GemmEx/F32 diagnostic. Full width/layout is fixed.
Outputs are reset with alternating byte patterns at beta0 to detect stale-output
dependence. Full BF16 output dumps, repeat drift, input immutability and guards
are checked. Native diagnostics need not be exact to complete; mismatch counts
are evidence, not success certificates for compatibility.

After explicit sole ownership is granted, from the repository root:

```sh
/opt/conda310/bin/python -B crates/ferrule-backend/tests/numeric_bf16_gemv/contract_run.py target/validation/capital-gemv-contract-20261006 hf-plain --execute-cuda --sole-owner-permission
```

Then run the same command with `hf-logged`, `native-129`, and `native-124`, in
that order. The runner rejects missing permission before GPU queries, verifies
all pins, requires all8 GPUs idle before each launch, authenticates process
custody, and records three idle observations afterward. Fresh receipt/output
paths are required; never overwrite old evidence. No such GPU command has run.

## Interpretation / backend decision

- If matched12.4 GemmEx fixes all cases but direct GEMV does not, the bounded
  next implementation is a general explicit GemmEx call-contract path, with
  its workspace/provider ownership accounted for and full114-case regression.
- If12.9 is exact but matched12.4 is not, report a toolchain/provider arithmetic
  incompatibility; do not call it a packing defect or pick tiles by expert ID.
- If repeated native BF16 outputs drift, preserve all hashes and investigate
  reproducibility/atomic behavior before claiming a deterministic backend fix.
- If observed metadata differs, stop and correct the **test contract**, not
  oracle values or production arithmetic.

No production patch is justified yet. Model dispatch remains untouched.
