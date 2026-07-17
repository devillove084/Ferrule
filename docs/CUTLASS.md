# GB10 CUTLASS/CuTe provider

Ferrule's production CUDA provider is deliberately narrow:

```text
NVIDIA GB10
compute capability 12.1
sm_121a
CUTLASS/CuTe 4.6.1
```

Unsupported hardware, missing plans, and unsupported artifact shapes fail explicitly. There is no CPU or generic CUDA production fallback.



## Ownership boundary

Rust remains the unique owner of:

- CUDA contexts and streams;
- allocations and graph-stable workspaces;
- the executable model plan;
- paged KV and sequence transactions;
- expert residency, I/O, and scheduling.

`crates/ferrule-cuda/native/cutlass/ferrule_cutlass.h` is a versioned C POD ABI. No C++ object crosses it. Every launch receives Ferrule-owned pointers and a Ferrule-owned stream. The provider allocates nothing and performs no host synchronization.

## Semantic ABI

The production FFI unit is a semantic superkernel, not a generic GEMM. ABI version 7 publishes eight operations:

| Operation | Fused boundary |
|---|---|
| FP8 QueryA + KV | one packed activation producer, two projection consumers |
| BF16 compressor | one F32→BF16 activation tile, two projections |
| HC producer | HC mix/split → pre-RMSNorm → FP8/E8M0 pack |
| shared FFN | gate/up → SwiGLU → hidden pack → down |
| routed MXFP4 MoE | stable-frame resolve → gate/up → pack → down |
| MLA output | OutputA → BF16 latent boundary → FP8 pack → OutputB |
| DSpark main projection/norm | target-tap FP8 projection → BF16 boundary → RMSNorm |
| DSpark hybrid MLA attention | committed paged context + ephemeral full-block KV → sink-aware tensor-core QK/softmax/PV |

Model plans bind one operation per semantic role. Small-M and tiled schedules are provider-private; model plans do not contain M=1/2/4/8 kernel variants. The semantic entry supports cross-tile M and validates its real grid/resource range.

CUTLASS 4.6.1 and CuTe provide MMA atoms, layouts, block-scaled types, and copy primitives. Ferrule implements the model-specific fused dataflow around those primitives.

## Reproducible dependency setup

The integration pins:

```text
repository  https://github.com/NVIDIA/cutlass.git
tag         v4.6.1
commit      e05f953a5b3d38adc240df2ff928e0421c2abba3
```

Fetch and verify it with:

```bash
just cutlass-setup
```

The checkout is stored under ignored build artifacts at `target/vendor/cutlass` by default. Set `FERRULE_CUTLASS_DIR` to use another checkout; it must resolve to the same pinned commit.

All GB10 build/test/run recipes depend on `cutlass-setup`, so normal use does not require a manual clone:

```bash
just build-cuda
just test-cutlass-provider
just dsv4-prefill-parity
```

CUTLASS is header-only in this integration. `crates/ferrule-cuda/build.rs` stays offline and uses NVCC through `cc::Build` to compile `native/cutlass/bridge.cu` for `sm_121a`. Keeping network access out of Cargo build scripts preserves offline and reproducible builds.

A direct `cargo oxide build` that bypasses `just` must either run `just cutlass-setup` first or provide `FERRULE_CUTLASS_DIR`.

## Validation contract

A provider operation is usable only when all of the following hold:

- ABI and manifest versions match;
- the compiled and runtime target is GB10 / SM121a;
- required pointers are non-null and correctly aligned;
- tensor shapes and quantization layouts match the model artifact;
- caller-owned workspaces have exact required capacity;
- the operation passes numerical parity for small and cross-tile M;
- the complete model path wins end to end.

Current validation includes provider ABI tests, dynamic-M tests including 4,097 rows, FP8/BF16 smoke tests, MXFP4 MoE smoke tests, and 43-layer packed-vs-token-loop CUDA parity.

## Benchmark rule

CUTLASS is not assumed faster merely because a kernel uses Tensor Cores. For every semantic operation:

1. measure the complete fused operation, including packs and epilogues;
2. sweep M across 1/2/4/8 and at least one cross-tile shape;
3. record workspace bytes and steady-state allocations;
4. verify graph safety and stream ownership;
5. run numerical and near-tie parity;
6. accept the change only when the complete 43-layer workload improves.

Microbenchmark wins that regress the resident verification sweep are removed.
