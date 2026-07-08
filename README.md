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
  <a href="docs/ROADMAP.md">Roadmap</a> ·
  <a href="docs/storage-residency-architecture.md">Storage</a>
</p>

---

## Current milestone

Ferrule just crossed an important line: **real local DeepSeek V4 Flash + DSpark
weights can enter an interactive CUDA chat loop from Rust**.

This is not a mock path and not a DeepSeek-specific CUDA fork. The DSV4 path
keeps the split we want long-term:

```
local HF safetensors → semantic source binding → DSV4 model semantics
  → generic CUDA source operators → readline chat
```

What works today:

- **DeepSeek V4 Flash + DSpark local CUDA chat** through `ferrule chat` / `just dsv4-chat`.
- Full 43-layer DSV4 greedy generation over local HF shards with strict `cuda` backend.
- **DSV4 benchmark can run through the latest resident runtime spine** via `bench-interactive --runtime-driver`: `GenerateRequest -> ResidentScheduler -> SchedulerAction -> ExecutionBatch -> TopKModelRunner -> ExecutionOutput -> token events -> finish/free`.
- Official DSV4 chat prompt wrapper (`<｜begin▁of▁sentence｜><｜User｜>...<｜Assistant｜></think>`).
- DSV4 prefill semantics: HC/layer traversal, window KV, compressed KV, indexer KV.
- Generic CUDA source operators for F32/BF16/FP8/FP4 linears, sparse attention,
  grouped output projection, HC pre/post/head, shared SwiGLU FFN, routed FP4
  experts, and lm_head top-k.
- **Storage/residency vocabulary** (`ferrule-runtime::storage` module):
  `StorageObjectId`, `Placement`, `ObjectLocator`, `ObjectReplica`,
  `TransferEngine` trait — the foundation for expert host-staging,
  frequency-aware eviction, and future async/remote backends.
- **`ExpertResidencyBackend` trait**: unifies the duplicated load/evict loop
  between CPU and CUDA expert paths. One `apply_streaming_step()` function
  replaces two inline loops.
- OLMoE remains the CUDA executable regression fixture.

A real local smoke from the current DSV4 CUDA chat milestone:

```
$ just dsv4-chat tokens=128
→ CUDA chat via cargo oxide build (arch: sm_121, tokens: 128)

DeepSeek-V4: 4096d×43L, 256e top-6, vocab=129280
  (cuda, attention=MLA, source=safetensors, arch=deepseek_v4)
Chat ready. Type /exit or Ctrl-D to quit.

You> hi
Ferrule> H! How can I help you?
stats> prefill=65789.6ms decode=41575.4ms pos=14
```

Next gates: **official numeric parity first, then 20 tok/s load-excluded decode**.

---

## Core features

| Feature | Why it matters |
|---|---|
| **Policy-composed Transformer runtime** | Model families map source tensors into semantic attention, FFN/MoE, KV, residency, quantization, and speculation policies — no per-family runner forks. |
| **MoE-first execution** | Router logits, hash/top-k selection, selected experts, shared experts, and expert residency are first-class runtime objects. |
| **Storage/residency vocabulary** | `StorageObjectId`, `ObjectLocator`, `Placement`, `ObjectReplica` — one typed vocabulary for all loadable/resident state. Expert eviction is frequency-aware, not just LRU. |
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
FERRULE_CUDA_ARCH=sm_121 just build-cuda
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
| Real large-model milestone | DeepSeek V4 Flash + DSpark full 43-layer CUDA greedy chat plus `bench-interactive --runtime-driver` through `ResidentTopKDriver` |
| DSV4 execution boundary | `ferrule-model::models::deepseek_v4` owns concrete HC, MLA, compressor/indexer, router, MoE, output semantics; runtime consumes `TopKModelRunner`/capability traits |
| Expert streaming | `ExpertStreamingPlanner` + `ExpertStreamingReader` + `ExpertResidencyBackend` trait |
| Storage/residency | `ferrule-runtime::storage` module: vocabulary types + `StorageCatalog` + `TransferEngine` traits |
| Attention | OLMoE GQA executable; DSV4 MLA sparse attention correctness-first CUDA path |
| Hyper-connections | DSV4 HC source binding + reference `hc_pre`/`hc_post`/`hc_head` |
| KV cache | `KvCache` trait, contiguous per-session, radix prefix cache |
| Quantization | Q4_0/Q8_0, FP4 E2M1 + E8M0 scales, FP8 E4M3FN, mixed precision policy |
| WeightPack | mmap'd reader, streaming writer, zero-copy slices, WeightPack-only load path |
| Runtime graph | `ferrule-runtime::graph` opaque IR, `GraphProgram`, `BackendObjectStore`, `ReferenceGraphBackend` |
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

1. **Official DSV4 parity** — tokenizer/template JSON parity, first-token
   golden tests, intermediate layer dumps.
2. **Device-resident decode** — remove host `Vec<f32>` boundaries; decode arena,
   GPU-resident KV, expert residency.
3. **Storage/residency integration** — wire `ferrule-runtime::storage` vocabulary
   into execution: CUDA `ExpertResidencyBackend`, host-staged expert cache,
   frequency-aware eviction.
4. **Graph bridge** — coarse `transformer_layer` execution through
   `ReferenceGraphBackend`, then split into fine-grained semantic ops.
5. **Benchmark contract** — DSV4 JSON bench with load/prefill/decode split,
   tok/s, bytes moved, resident experts.

---

## Documentation

| Document | Content |
|---|---|
| [Architecture](docs/ferrule_arch.md) | Crate layout, runtime, graph, storage, model-family boundary, GPU kernels, DSV4 execution |
| [Roadmap](docs/ROADMAP.md) | Priority roadmap P0–P8, implementation order, gap matrix, model bring-up contract |
| [Storage & Residency](docs/storage-residency-architecture.md) | `StorageObjectId`, `Placement`, `ObjectLocator`, `ObjectReplica`, `TransferEngine`, `ExpertResidencyBackend` — full design |
| [Runtime Graph](docs/runtime-graph-architecture.md) | Graph IR, `GraphNode`, `GraphBackend`, `ExecutionBatch`, `ExternalBindingPlan` |

---

## License

Apache-2.0
