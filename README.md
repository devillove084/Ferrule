<h1 align="center">Ferrule</h1>

<p align="center">
  <strong>Rust-native LLM runtime for sparse MoE, expert streaming, and hardware-aware execution.</strong>
</p>

<p align="center">
  Router decisions, selected experts, quantized weights, KV cache, expert residency,
  and storage objects are explicit runtime concepts — not hidden behind opaque kernels.
</p>

<p align="center">
  <img alt="Rust" src="https://img.shields.io/badge/Rust-native-f97316?style=flat-square" />
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-cuda--oxide-22c55e?style=flat-square" />
  <img alt="MoE" src="https://img.shields.io/badge/MoE-router%20%2B%20top--k%20experts-8b5cf6?style=flat-square" />
  <img alt="WeightPack" src="https://img.shields.io/badge/WeightPack-execution%20artifact-2563eb?style=flat-square" />
  <img alt="Storage" src="https://img.shields.io/badge/Storage-residency%20vocabulary-06b6d4?style=flat-square" />
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#current-milestone">Milestone</a> ·
  <a href="#core-features">Features</a> ·
  <a href="docs/ferrule_arch.md">Architecture</a> ·
  <a href="docs/execution-engine-architecture.md">Execution Engine</a> ·
  <a href="docs/ROADMAP.md">Roadmap</a> ·
  <a href="docs/storage-residency-architecture.md">Storage</a>
</p>

---

## Current milestone

Ferrule has a real local CUDA DeepSeek-V4 Flash + DSpark path on GB10. The current
milestone includes native packed decode, ragged prefill, mixed batches, and physical
paged multi-plane KV. Token-loop execution is retained only as an explicit
page-managed differential oracle.

Verified on `models/DeepSeek-V4-Flash-DSpark` with `sm_121a`:

- three independent 43-layer parity runs have every layer `max_abs_diff = 0.0`;
- cut top-1 values at layers 1/5/23/43 are `83484/83484/108527/30594`;
- cold prefill→decode continuation repeatedly emits `[30594, 1175]`;
- dynamic RoPE and combined-KV growth have CUDA regressions at their capacity
  boundaries;
- full 43-layer greedy chat/generation, FP4 Tensor Core MoE, GPU compressor
  numerics, and deterministic route-ranked reduction are active.

E1–E5 are complete. Ferrule now has a dependency-neutral `ExecutionBatch`, prepared
plans and persistent arenas, explicit per-sequence state, token-budgeted native
multi-session execution, and authoritative runtime/CUDA page transactions. Exact
prefixes share physical pages with partial-tail COW; rollback and preempt/restore
replace model-local CUDA checkpoint copies.

The architecture is **not** yet vLLM/SGLang-class serving:

- router and expert residency remain host controlled;
- DSV4 uses device-resident eager execution but has no stable graph buckets yet;
- device sampling, fusion, radix lookup policy, and an active server command remain
  future work.

Next are E6 device routing/runtime residency, E7 stable graph buckets, and E8 fusion,
device sampling, and competitive validation. See the
[execution engine design](docs/execution-engine-architecture.md) and
[roadmap](docs/ROADMAP.md).

---

## Core features

| Feature | Why it matters |
|---|---|
| **Policy-composed model descriptions** | Model families map source tensors into semantic attention, FFN/MoE, KV, residency, quantization, and speculation policies. DSV4 now separates fully prepared layers, backend expert runtime, explicit sequence state, and reusable eager arenas. |
| **Neutral execution ABI** | One public `ExecutionBatch` expresses packed/ragged rows, phases, state slots, KV bindings, and strict logits intent; runtime correlation stays private and native multi-session execution consumes it directly. |
| **MoE-first execution** | Router logits, hash/top-k selection, selected experts, shared experts, and DSV4 expert-streaming mechanisms are explicit objects rather than opaque kernel details. Cross-request residency coordination is still planned. |
| **Storage/residency vocabulary** | `StorageObjectId`, `ObjectLocator`, `Placement`, `ObjectReplica` provide typed storage vocabulary. DSV4 currently uses a model-local planner; runtime-owned budget/eviction coordination is an E6 target. |
| **cuda-oxide kernels** | Custom CUDA kernels integrated with the Rust runtime: quantized GEMV, packed FP4 expert execution, sparse attention, artifact-preserving operators. |
| **Safetensors source binding** | Inspect and bind Hugging Face safetensors by semantic role, with bounded reads instead of loading a 100 GB+ checkpoint into RAM. |
| **WeightPack execution artifact** | Layer weights quantized once and reloaded from a Ferrule-owned package. GGUF remains a compatibility/PK path. |
| **Runtime graph IR** | Opaque graph with semantic ops, typed artifact bindings, shape registry, and backend object store. Model-family names stay out of graph nodes. |
| **CPU/reference anchors** | CPU reference pieces validate CUDA kernels, source-format decoders, router behavior, and HC math — without a legacy full-model CPU runner. |
| **Edge/hardware direction** | Expert placement, streaming, WeightPack layout, and scheduling adapt to VRAM, DRAM, NVMe, and future multi-GPU / multi-node cooperation. |

---

## System vision

Ferrule is designed around a simple idea: future LLM systems need to co-design
model structure, runtime state, and hardware placement.

<p align="center">
  <img src="docs/assets/ferrule-current-architecture.svg" alt="Ferrule architecture" width="100%" />
</p>

Near term: llama.cpp-level local usability with a more explicit runtime
architecture — fast cached startup, sampling controls, templates, quality
checks, benchmarks, future local serving, and source-preserving bring-up for
mainstream model families.

Long term: a runtime substrate for edge-cloud LLM systems:

- cloud builds model versions, WeightPack artifacts, calibration data, adapters
- edge devices run private low-latency inference and collect rollout traces
- router statistics guide expert placement, prefetch, and offload
- KV/session state becomes movable and eventually distributed
- speculation modules (DSpark/MTP) attach through a target/draft policy
- DP/TP/EP/SP/CP/PP placement evolves under one state-aware runtime

---

## Quick start

The fastest path is through `justfile` wrappers. For CUDA, prefer `just` /
`cargo oxide` commands; plain `cargo test -p ferrule-cuda` can miss cuda-oxide
artifact wiring.

1. **Check the environment:**

```bash
just cuda-info
just oxide-doctor
```

2. **Build the CUDA release binary:**

```bash
just build-cuda

# Override architecture if auto-detection fails:
FERRULE_CUDA_ARCH=sm_121a just build-cuda
```

3. **Put the local DSV4 source checkout here:**

```
models/DeepSeek-V4-Flash-DSpark
```

4. **Run real local DeepSeek V4 Flash + DSpark CUDA chat:**

```bash
just dsv4-chat tokens=128
```

Inside chat:

```
/reset    clear session state
/stats    show session stats
/experts  show DSV4 layer/cache stats
/ctx      show context window usage
/exit     quit
```

5. **One-shot DSV4 CUDA smoke:**

```bash
just dsv4-cuda-generate Hello 2 4096 --chat
```

6. **OLMoE regression fixture (if present):**

```bash
just chat models/OLMoE-Instruct q4 -n 256
```

---

## Capability map

| Area | Status |
|---|---|
| Executable model fixture | OLMoE sparse MoE CUDA path via `GpuOlmoeRunner` |
| Real large-model milestone | DeepSeek V4 Flash + DSpark full 43-layer CUDA greedy chat plus `bench-interactive` through `ResidentTopKDriver` |
| Execution ABI | E1–E4 complete: public `ExecutionBatch`, prepared-executor traits, runtime-private `ScheduledBatch`, strict validation, and native packed multi-session execution |
| DSV4 execution boundary | The model owns HC/MLA/compressor/router/MoE math; runtime remains model-neutral and owns scheduling plus logical page lifecycle through generic traits |
| Expert streaming | `ExpertStreamingPlanner` + `ExpertStreamingReader` + `ExpertResidencyBackend` trait |
| Storage/residency | `ferrule-runtime::storage` module: vocabulary types + `StorageCatalog` + `TransferEngine` traits |
| Attention | OLMoE GQA executable; DSV4 MLA sparse attention correctness-first CUDA path |
| Hyper-connections | DSV4 HC source binding + reference `hc_pre`/`hc_post`/`hc_head` |
| KV cache | E5 complete: `KvPageManager` plus bounded CUDA multi-plane pools, paged attention/indexer kernels, rollback, COW, preempt/restore, and exact-prefix sharing; host caches are reference infrastructure |
| Quantization | Q4_0/Q8_0, FP4 E2M1 + E8M0 scales, FP8 E4M3FN, mixed precision policy |
| WeightPack | mmap'd reader, streaming writer, zero-copy slices, WeightPack-only load path |
| Runtime graph | `ferrule-runtime::graph` opaque IR, `GraphProgram`, `BackendObjectStore`, and `ReferenceGraphExecutor::execute` with caller-owned `ReferenceGraphSequenceState` |
| Sampling | temperature, top-k, top-p, min-p, repeat penalty, seed, stop strings, logprobs |
| Chat templates | OLMoE, ChatML, Llama3, Qwen, DeepSeek-V4, Plain |
| Serving | design target; no active CLI server command in the current 5-crate workspace |
| Structured decoding | token mask API, program-like generation API |
| Speculation | DSpark/MTP metadata represented as generic speculation attachment policy |
| Training/RL | design target, not implemented |

---

## Useful commands

### Environment and build

```bash
just cuda-info          # Show GPU/arch detection
just oxide-doctor       # cuda-oxide environment check
just build              # Auto-detect: CUDA if available, CPU otherwise
just build-cuda         # Explicit CUDA build
just build-cpu          # CPU-only build
just check              # Quick check
```

### DeepSeek V4 / DSpark

```bash
just dsv4-chat tokens=128                        # Interactive chat
just dsv4-runtime-driver-bench                   # bench-interactive via ResidentTopKDriver
# positional override: prompt1 prompt2 tokens warmup chunk layers
just dsv4-runtime-driver-bench "Hello" "Explain Ferrule in one sentence." 1 0 2 43
just test-dsv4-runtime-driver-local              # opt-in ignored local runtime-driver test
just dsv4-cuda-generate Hello 2 4096 --chat       # One-shot generation
just dsv4-cuda-first-token Hello 1               # First-token diagnostic
just dsv4-cuda-probe "one two three" 3 1 0       # Layer-limited probe
just dsv4-parity-json "Who are you?" output.json # Tokenizer parity JSON
```

### Generic CLI surface

```bash
cargo run -p ferrule-cli -- info models/OLMoE-Instruct
just chat models/OLMoE-Instruct q4 -n 256        # Interactive chat wrapper
cargo run -p ferrule-cli -- cuda                 # CUDA probe + smoke benchmark
cargo run -p ferrule-cli -- inspect-weightpack path/to/model.weightpack
cargo run -p ferrule-cli -- expert-stream-smoke models/OLMoE-Instruct --layer 0 --expert 0
```

### Validation

```bash
just test           # All tests
just test-graph     # Graph IR tests
just test-runtime   # Runtime tests
just test-cuda      # CUDA tests (via cargo oxide)
just fmt            # Format check
just clippy         # Lint
just lint           # fmt + clippy
```

---

## Active development focus

1. **Implement E4 native ragged multi-session execution** — consume the completed
   prepared-plan, per-sequence state, and exact-bucket arena ownership without
   reintroducing a default-sequence swap path.
2. **Make batching physical in E5** — bind ragged execution to physical multi-plane
   paged KV before claiming continuous batching, prefix reuse, or preemption.
3. **Move routing and residency policy into E6** — distinguish fixed prepared
   resources from dynamic expert-generation changes and coordinate them across
   active sessions.
4. **Build stable E7 graph buckets** — lower the unified allocation-free eager
   stages into reusable graphs; do not restore the deleted token-specific one-shot
   graph path.
5. **Fuse and benchmark in E8** — add device sampling, fusion, and competitive GB10
   comparisons only after E4–E7 stabilize the serving execution shape.

---

## Documentation

| Document | Content |
|---|---|
| [Architecture](docs/ferrule_arch.md) | Repository crates, model boundaries, current runtime, serving direction, and alignment targets |
| [Execution Engine](docs/execution-engine-architecture.md) | E1–E5 implemented ownership: neutral ABI, prepared plans, sequence state, arenas, native batching, and physical paged KV; E6–E8 targets follow |
| [Roadmap](docs/ROADMAP.md) | Canonical E0–E8 dependency plan with phase-owned correctness/performance gates and non-goals |
| [Storage & Residency](docs/storage-residency-architecture.md) | Current/target expert residency ownership, slots, leases, transfers, policy, metrics, and E6 migration |
| [Runtime Graph](docs/runtime-graph-architecture.md) | Device-independent graph IR, `GraphNode`, dialects, `GraphProgram`, external bindings, and explicit reference execution state |

---

## License

Apache-2.0
