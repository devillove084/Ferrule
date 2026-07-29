# Current GB10 SM121 CUTLASS/CuTe provider

Ferrule's current production kernel implementation is deliberately narrow. It is the
SM121 provider for the DeepSeek-V4/GB10 validation profile, not Ferrule's global hardware
or core runtime contract:

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

This ownership boundary does not change when additional providers are introduced.
`crates/ferrule-cuda/native/cutlass/ferrule_cutlass.h` is the current SM121 provider's
versioned C POD ABI. No C++ object crosses it. Every launch receives Ferrule-owned pointers
and a Ferrule-owned stream. The provider allocates nothing and performs no host
synchronization.

The future provider-neutral contract, compatibility rules, and adapter requirements are
specified in [Kernel Provider ABI](KERNEL_PROVIDER_ABI.md). [Architecture](ARCHITECTURE.md)
describes how providers fit beneath the Rust-owned runtime.

## Semantic ABI

The current SM121 FFI unit is a semantic superkernel, not a generic GEMM. The checked-in
code ABI is **9** and publishes ten operations:

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
| DSpark proposal head | HC head/final norm → BF16 LM projection → sequential Markov selection and confidence |
| FP8 projection | one packed activation producer, one projection consumer |

Model plans bind one operation per semantic role. Small-M and tiled schedules are provider-private; model plans do not contain M=1/2/4/8 kernel variants. The semantic entry supports cross-tile M and validates its real grid/resource range.

For this GB10 provider, CUTLASS 4.6.1 and CuTe provide MMA atoms, layouts, block-scaled
types, and copy primitives. Ferrule implements the model-specific fused dataflow around
those primitives. CUTLASS 4.6.1 is the pinned implementation dependency for the current
GB10 profile; it is not a Ferrule-wide core or kernel-provider ABI contract.

## Current hard coupling and adapter migration

The current path is not yet a provider-neutral build/load adapter. SM121 details remain
hard-coupled in:

- `crates/ferrule-cuda/build.rs`, which validates `sm_121a` and directly compiles the
  CUTLASS translation unit;
- `crates/ferrule-cuda/native/cutlass/bridge.cu`, which includes the SM121 kernels,
  constructs their manifest, and exports their launch functions;
- the mirrored kernel IDs in `crates/ferrule-cuda/src/cutlass.rs` and
  `crates/ferrule-cuda/native/cutlass/ferrule_cutlass.h`, plus the operation-to-ID mapping
  in `crates/ferrule-cuda/src/provider.rs`;
- `ferrule_cutlass.h` itself, which currently combines the C ABI structs with the SM121
  kernel catalog.

The gradual migration in the provider ABI specification **has not been implemented**. It
first freezes and inventories the ABI9 direct functions, then introduces an SM121 adapter
that maps the neutral manifest/prepare/launch contract to those functions. One operation
will move through shadow comparison and a temporary observable fallback before the same
process expands to the remaining operations and removes the direct exports. Provider-specific
build and bridge selection can then be separated without changing Rust ownership.

Until those stages land, the runtime continues to use the directly wired SM121 path. This
document does not claim a generic plugin, portable adapter, or multi-provider implementation;
additional hardware can be registered only after its own validation and performance
contract passes.

### Portable adapter migration exit gate

Throughout migration—and before the portable adapter migration is considered complete or
any ABI9 direct fallback or export is removed—all of the following must hold:

- all ten published ABI9 operations retain direct-vs-adapter per-operation parity;
- the complete 43-layer packed path retains its established parity;
- the proposal path passes the near-tie and acceptance corpus without semantic drift;
- `status_i32` and `route_error` preserve their existing status, ordering, and failure
  semantics;
- prepare, launch, capture, and replay introduce no implicit host/device allocation or
  synchronization;
- provider DSO, CUDA module, plan, and graph lifetimes and unload ordering follow
  [Kernel Provider ABI](KERNEL_PROVIDER_ABI.md).

These are migration gates, not claims about the current directly wired implementation or
unfinished adapter.

## Reproducible dependency setup

The current GB10 implementation pins:

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
just dsv4-runtime-driver-bench
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
