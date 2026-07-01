<h1 align="center">Ferrule</h1>

<p align="center">
  <strong>Rust-native, composable LLM inference for sparse MoE, expert streaming, and hardware-aware execution.</strong>
</p>

<p align="center">
  Ferrule keeps model-family policies, tensor source bindings, router/top-k experts, quantized weights, KV cache, and future speculation/parallelism state as explicit runtime concepts.
</p>

<p align="center">
  <img alt="Rust" src="https://img.shields.io/badge/Rust-native-f97316?style=flat-square" />
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-cuda--oxide-22c55e?style=flat-square" />
  <img alt="MoE" src="https://img.shields.io/badge/MoE-router%20%2B%20top--k%20experts-8b5cf6?style=flat-square" />
  <img alt="WeightPack" src="https://img.shields.io/badge/WeightPack-source%20aware-2563eb?style=flat-square" />
  <img alt="Edge" src="https://img.shields.io/badge/edge--cloud-state%20aware-06b6d4?style=flat-square" />
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#current-milestone">Milestone</a> ·
  <a href="#core-features">Features</a> ·
  <a href="docs/ferrule_arch.md">Architecture</a> ·
  <a href="docs/ROADMAP.md">Roadmap</a>
</p>

---

## Current milestone

Ferrule has two active tracks:

1. **Executable OLMoE path** — Ferrule can run **OLMoE-1B-7B-0924-Instruct** end-to-end with a real sparse MoE path:

```text
safetensors → Rust loader → WeightPack/Q4_0 cache → cuda-oxide kernels → router/top-k experts → chat/server
```

2. **Generic model bring-up path** — Ferrule is being refactored around semantic model-family policies and source bindings, with **DeepSeek V4 Flash + DSpark** as the current pressure-test model:

```text
HF safetensors inventory → family tensor descriptors → source payloads → generic attention / HC / router / expert policies
```

What works today:

- CPU FP32 reference inference and GPU Q4_0 inference for the current OLMoE path.
- WeightPack cache support for quantized OLMoE startup and inspection.
- Interactive chat REPL, one-shot run, benchmark, compare-logits, perplexity, and minimal OpenAI-compatible server commands.
- Generic `ModelSupportContract` / `EnginePlan` policy skeleton for non-OLMoE bring-up.
- DeepSeek V4 Flash + DSpark local HF inventory parsing: 72,317 tensors, 48 shards, ~166.9GB source, with semantic tensor classification.
- Source-preserving expert streaming from local HF shards, including packed FP4 routed expert tensor sets.
- Real DSV4 router/shared-expert bindings and CPU/reference MoE fixture coverage.
- Real DSV4 attention source binding for core MLA tensors plus compressor/indexer auxiliary slices.
- Real DSV4 Hyper-Connection source binding for layer HC and global HC head tensors, with reference `hc_pre` / `hc_post` / `hc_head` primitives.
- Initial CUDA Oxide correctness-oriented kernels for packed FP4 experts and sparse attention ABI work.

Ferrule is no longer just an OLMoE runner: OLMoE remains the executable regression fixture, while the main architecture is moving toward a generic, policy-composed Transformer runtime.

---

## Core features

| Feature | Why it matters |
|---|---|
| **Policy-composed Transformer runtime** | Model families map source tensors into semantic attention, FFN/MoE, KV, residency, quantization, and speculation policies instead of forking runners per model. |
| **MoE-first execution** | Router logits, hash/top-k selection, selected experts, shared experts, and expert residency are first-class runtime objects. |
| **Rust-native model runtime** | Model metadata, source tensor slices, WeightPack files, CUDA buffers, KV cache, sessions, and scheduler state have explicit ownership and typed boundaries. |
| **cuda-oxide kernels** | Custom CUDA kernels stay integrated with the Rust runtime, enabling MoE-specific quantized GEMV, packed FP4 expert execution, sparse attention, and future fusion work. |
| **Safetensors source binding** | Ferrule can inspect and bind Hugging Face safetensors by semantic role, with bounded reads instead of loading a whole 100GB+ checkpoint into RAM. |
| **WeightPack execution artifact** | Layer weights can be quantized once and reloaded from a Ferrule-owned package/cache; GGUF remains a compatibility/PK path rather than the only source format. |
| **CPU/reference path** | CPU reference execution anchors CUDA kernels, source-format decoders, router behavior, HC math, and future model support. |
| **State-aware design** | KV pages, source artifacts, WeightPack artifacts, model versions, adapters, speculation state, and future rollout/checkpoint state are planned as managed runtime state. |
| **Edge/hardware direction** | Expert placement, streaming, WeightPack layout, and scheduling can adapt to VRAM, DRAM, NVMe, cloud artifacts, and future multi-GPU / multi-node / RISC-V/GPU/NPU cooperation. |

---

## System vision

Ferrule is designed around a simple idea: future LLM systems need to co-design model structure, runtime state, and hardware placement.

<p align="center">
  <img src="docs/assets/ferrule-current-architecture.svg" alt="Ferrule system vision" width="100%" />
</p>

Near term, Ferrule aims to reach llama.cpp-level local usability while keeping a more explicit runtime architecture: fast cached startup, sampling controls, templates, quality checks, benchmarks, a small local server, and source-preserving bring-up for mainstream model families.

Long term, Ferrule should become a runtime substrate for edge-cloud LLM systems:

- cloud builds model versions, weight pack artifacts, calibration data, and adapters
- edge devices run private low-latency inference and collect rollout traces
- router statistics guide expert placement, prefetch, and offload
- KV/session state can become movable and eventually distributed
- speculation modules such as DSpark/MTP can attach through a target/draft policy
- hardware counters feed back into quantization and scheduling policies
- DP/TP/EP/SP/CP/PP placement can evolve under one state-aware runtime
- RISC-V/GPU/NPU paths can cooperate under one state-aware runtime

---

## Quick start

Build CPU-only:

```bash
cargo build --release
```

Build CUDA with cuda-oxide:

```bash
cargo oxide build --features cuda --arch sm_86
```

Download the current executable OLMoE-Instruct fixture with the project helper:

```bash
./.venv/bin/python scripts/download_ms.py \
  LLM-Research/OLMoE-1B-7B-0924-Instruct \
  --out models

ln -s LLM-Research/OLMoE-1B-7B-0924-Instruct models/OLMoE-Instruct
```

Run OLMoE chat:

```bash
./target/release/ferrule chat models/OLMoE-Instruct -q q4 -n 256
```

CPU reference chat:

```bash
./target/release/ferrule chat models/OLMoE-Instruct -q cpu -n 128
```

Inspect a local DeepSeek V4 Flash + DSpark source checkout if present:

```bash
./target/release/ferrule info models/DeepSeek-V4-Flash-DSpark

./target/release/ferrule expert-stream-smoke \
  models/DeepSeek-V4-Flash-DSpark \
  --layer 0 \
  --expert 0 \
  --max-slice-mb 64
```

---

## Current capability map

| Area | Status |
|---|---|
| Executable model fixture | OLMoE-style sparse MoE, CPU FP32 and GPU Q4_0 paths |
| Generic bring-up target | DeepSeek V4 Flash + DSpark source inventory and semantic source binding |
| Inference commands | `run`, `gpu-run`, `chat`, `server`, `bench-infer`, `compare-logits`, `perplexity` |
| MoE execution | router → top-k/hash selection → routed experts → optional shared FFN; CPU/reference DSV4 MoE fixtures pass |
| Expert streaming | bounded local HF shard reads for selected experts; source FP4 expert bundles and resident handle abstractions exist |
| Attention | OLMoE GQA executable path; DSV4 attention source binding, sparse attention reference, and CUDA sparse-attention ABI in progress |
| Hyper-Connections | DSV4 HC source binding plus reference `hc_pre` / `hc_post` / `hc_head` primitives |
| KV cache | contiguous, multi-session, paged, prefix, and radix-cache components exist; GPU OLMoE path uses persistent session KV |
| Quant/cache artifact | WeightPack cache for quantized weights; source-preserving FP4/FP8 decoders for DSV4 bring-up |
| Sampling/control | greedy and configurable sampling arguments, stop handling, structured/program-like generation utilities |
| Serving | minimal OpenAI-compatible local server path |
| Speculation | DSpark/MTP metadata and tensor roles are represented as a generic speculation attachment policy |
| Training/RL | design target, not implemented yet |

---

## Useful commands

```bash
# OLMoE metadata
./target/release/ferrule info models/OLMoE-Instruct

# DeepSeek V4 / DSpark source metadata and tensor policy summary
./target/release/ferrule info models/DeepSeek-V4-Flash-DSpark

# One-shot CPU generation on the current executable fixture
./target/release/ferrule run models/OLMoE-Instruct -p "Paris is" -n 16

# One-shot GPU generation on the current executable fixture
./target/release/ferrule gpu-run models/OLMoE-Instruct -p "Paris is" -n 16 -q q4

# Minimal local server
./target/release/ferrule server models/OLMoE-Instruct -q q4 --host 127.0.0.1 --port 8080

# Compare CPU FP32 vs GPU quantized logits
./target/release/ferrule compare-logits models/OLMoE-Instruct -p "Paris is" -n 16 -q q4

# Benchmark prompt/decode throughput, excluding model-load time
./target/release/ferrule bench-infer models/OLMoE-Instruct -p "Paris is" -n 64 -q q4

# Smoke-test one source-preserved DSV4 routed expert from local HF shards
./target/release/ferrule expert-stream-smoke models/DeepSeek-V4-Flash-DSpark --layer 0 --expert 0 --max-slice-mb 64

# CUDA probe and GEMV benchmark
./target/release/ferrule cuda

# Enable token top-k debug logging
FERRULE_DEBUG_TOPK=1 ./target/release/ferrule chat models/OLMoE-Instruct -q q4 -n 32
```

Useful validation commands:

```bash
cargo test -p ferrule-model
cargo test -p ferrule-runtime
cargo check -p ferrule-cli --features cuda
cargo oxide test -- -p ferrule-cuda
```

---

## Active development focus

Ferrule's current implementation focus is the generic DeepSeek V4 vertical slice while keeping OLMoE as the executable regression fixture:

1. **Layer source binding** — compose attention, HC, router, shared FFN, routed experts, and KV state into a typed per-layer bundle.
2. **DSV4 attention path** — execute the bound MLA/sparse attention payload through generic `AttentionPolicy` dispatch.
3. **Source-format CUDA** — continue validating FP8 attention/shared linears, packed FP4 experts, and sparse attention against CPU/reference math.
4. **Single-node DGX Spark smoke** — run the base DeepSeek V4 target path first; attach DSpark/MTP through generic speculation policy after target decode is stable.

---

## Documentation

- [Architecture](docs/ferrule_arch.md)
- [Roadmap and llama.cpp gap analysis](docs/ROADMAP.md)
- [Temporary inference roadmap / DeepSeek V4 bring-up notes](docs/TEMP_INFERENCE_ROADMAP.md)
- [Paper and system notes](docs/paper_reads.md)
