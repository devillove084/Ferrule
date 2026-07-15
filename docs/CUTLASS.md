# CUTLASS provider migration

Ferrule is CUTLASS-first for regular matrix compute, not CUTLASS-owned as a runtime.
Rust remains the unique owner of CUDA contexts, streams, allocations, execution plans,
paged KV, expert residency, sequence transactions, graphs, cancellation, and scheduling.
CUTLASS receives only versioned POD descriptors and Ferrule-owned pointers/workspaces.

## Provider model

```text
semantic operation + row bucket + numerical contract
  -> compatible provider manifests
  -> provider can_implement
  -> measured winner compiled into LayerKernelPlan
  -> direct launch on a Ferrule-owned stream
```

Provider families are selected by implemented kernel ISA, not product-name branches in
model code:

| Development target | Native provider direction | Portable fallback |
|---|---|---|
| RTX 3090, SM86 | Ampere BF16 mma.sync | F32 SIMT |
| H20/H200, SM90a | Hopper WGMMA BF16/FP8 | Ampere-compatible BF16 mma.sync |
| B200, SM100a | Blackwell tcgen05 BF16/FP8/FP4 | BF16 mma.sync where valid |
| GB10, SM121a | Blackwell consumer FP8/MXFP4 mma.sync | BF16/F32 providers |

NVCC compiles a configured CUTLASS kernel for a target, but it does not redesign an
Ampere kernel into an optimal Hopper or Blackwell schedule. Ferrule therefore carries a
small number of architecture-family specializations behind one semantic operation ID.
CUDA fatbins and CUTLASS `can_implement` handle compatibility; benchmark evidence selects
the production winner.

## Current ABI

`crates/ferrule-cuda/native/cutlass/ferrule_cutlass.h` defines the native ABI.
No C++ object crosses it. The provider currently publishes:

- `F32Simt`: portable `A * W^T` baseline;
- `Bf16MmaSync`: F32 input, BF16 weights, F32 output, with a caller-owned BF16
  activation-pack workspace.

The BF16 path performs no allocation or synchronization. Its pack and GEMM are enqueued
on the stream supplied by Ferrule. `CudaArtifactOperatorContext` exposes an explicit
BF16 rows entry point so execution images can own one workspace per row bucket.

## Migration order

1. BF16 compressor dual projection and other regular BF16 projections.
2. Batched BF16/FP8 verification head.
3. FP8 MLA QueryA/KV and QueryB bundles.
4. Shared FFN grouped GEMM.
5. Routed FP4 MoE grouped/segment GEMM.
6. Provider-native fused epilogues where profiling proves value.

Each migration keeps the cuda-oxide implementation as a numerical oracle and fallback
until all required row buckets pass parity and target-hardware A/B benchmarks.

## Machine workflow

Fetch the pinned CUTLASS 4.6.1 headers once per checkout:

```bash
just cutlass-setup
```

Build for the detected local GPU:

```bash
just build-cutlass
```

Or cross-compile explicitly:

```bash
just build-cutlass sm_86
just build-cutlass sm_90a
just build-cutlass sm_121a
```

Run the provider ABI and numerical smoke tests:

```bash
just test-cutlass-provider
```

## Benchmark rule

CUTLASS is not assumed faster merely because a kernel uses Tensor Cores. For every
semantic operation and rows=1/2/4/8 bucket, record:

- end-to-end operation time including required packs and epilogues;
- workspace bytes and steady-state allocation count;
- graph capture safety;
- numerical parity and near-tie behavior;
- complete-cycle impact rather than isolated TFLOP/s only.

A specialization becomes the default only when it beats the current provider under the
same correctness and execution contract.
