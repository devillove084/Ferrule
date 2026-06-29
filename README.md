<h1 align="center">Ferrule</h1>

<p align="center">
  <strong>Rust-native sparse MoE inference for edge GPUs and hardware-aware LLM systems.</strong>
</p>

<p align="center">
  Ferrule keeps router, top-k experts, quantized weights, KV cache, and future rollout/training state as explicit runtime concepts.
</p>

<p align="center">
  <img alt="Rust" src="https://img.shields.io/badge/Rust-native-f97316?style=flat-square" />
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-cuda--oxide-22c55e?style=flat-square" />
  <img alt="MoE" src="https://img.shields.io/badge/MoE-router%20%2B%20top--k%20experts-8b5cf6?style=flat-square" />
  <img alt="Q4_0" src="https://img.shields.io/badge/Q4__0-llama.cpp%20layout-2563eb?style=flat-square" />
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

Ferrule can run **OLMoE-1B-7B-0924-Instruct** end-to-end with a real sparse MoE path:

```text
safetensors → Rust loader → Q4_0 qcache → cuda-oxide kernels → router/top-k experts → chat
```

What works today:

- CPU FP32 reference inference
- GPU Q4_0 quantized inference
- llama.cpp-compatible Q4_0 packing/dequant layout
- OLMoE router semantics for `norm_topk_prob=false`
- explicit router → top-k experts → expert gate/up/down loop
- interactive chat REPL with persistent KV cache
- quantized layer cache: `model.q4_0_llama.qcache`
- Q8_0 quantizer and CUDA GEMV kernels are present for validation work

This is the first milestone where Ferrule can chat with an instruct MoE model while preserving the sparse expert execution path instead of flattening it into a dense abstraction.

---

## Core features

| Feature | Why it matters |
|---|---|
| **MoE-first execution** | Router logits, top-k selection, expert loops, and expert weights are first-class runtime objects. |
| **Rust-native model runtime** | Model metadata, qcache files, CUDA buffers, and KV cache have explicit ownership and typed boundaries. |
| **cuda-oxide kernels** | Custom CUDA kernels stay integrated with the Rust runtime, enabling MoE-specific quantized GEMV and router/expert fusion work. |
| **Safetensors-first loading** | Ferrule can load Hugging Face / ModelScope-style OLMoE shards directly instead of requiring a GGUF conversion step. |
| **Quantized qcache** | Layer weights can be quantized once and reloaded from an mmap-backed cache. qcache-only startup is the next target. |
| **CPU reference path** | CPU FP32 inference provides a correctness anchor for GPU quantization, router behavior, and future model support. |
| **State-aware design** | KV pages, qcache artifacts, model versions, LoRA adapters, trajectories, and checkpoints are planned as managed runtime state. |
| **Edge/hardware direction** | Expert placement, qcache layout, and scheduling can adapt to VRAM, DRAM, NVMe, cloud artifacts, and future RISC-V/GPU/NPU cooperation. |

---

## System vision

Ferrule is designed around a simple idea: future LLM systems need to co-design model structure, runtime state, and hardware placement.

<p align="center">
  <img src="docs/assets/ferrule-current-architecture.svg" alt="Ferrule system vision" width="100%" />
</p>

Near term, Ferrule aims to reach llama.cpp-level local usability for sparse MoE models: fast cached startup, sampling controls, templates, quality checks, benchmarks, and a small local server.

Long term, Ferrule should become a runtime substrate for edge-cloud LLM systems:

- cloud builds model versions, qcache artifacts, calibration data, and adapters
- edge devices run private low-latency inference and collect rollout traces
- router statistics guide expert placement, prefetch, and offload
- KV/session state can become movable and eventually distributed
- hardware counters feed back into quantization and scheduling policies
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

Download OLMoE-Instruct with the project helper:

```bash
./.venv/bin/python scripts/download_ms.py \
  LLM-Research/OLMoE-1B-7B-0924-Instruct \
  --out models

ln -s LLM-Research/OLMoE-1B-7B-0924-Instruct models/OLMoE-Instruct
```

Run chat:

```bash
./target/release/ferrule chat models/OLMoE-Instruct -q q4 -n 256
```

CPU reference chat:

```bash
./target/release/ferrule chat models/OLMoE-Instruct -q cpu -n 128
```

---

## Current capability map

| Area | Status |
|---|---|
| Model family | OLMoE-style sparse MoE |
| Inference paths | CPU FP32, GPU Q4_0, Q8_0 kernels present |
| MoE execution | router → top-k → expert gate/up/down loop |
| Chat | interactive REPL, OLMoE-Instruct template shortcut, EOS stop |
| KV cache | persistent single-session GPU KV cache |
| Quant cache | mmap-backed per-layer qcache for quantized weights |
| Sampling | greedy only today |
| Serving | CLI only today |
| Training/RL | design target, not implemented yet |

---

## Useful commands

```bash
# Model metadata
./target/release/ferrule info models/OLMoE-Instruct

# One-shot CPU generation
./target/release/ferrule run models/OLMoE-Instruct -p "Paris is" -n 16

# One-shot GPU generation
./target/release/ferrule gpu-run models/OLMoE-Instruct -p "Paris is" -n 16 -q q4

# CUDA probe and GEMV benchmark
./target/release/ferrule cuda

# Enable token top-k debug logging
FERRULE_DEBUG_TOPK=1 ./target/release/ferrule chat models/OLMoE-Instruct -q q4 -n 32
```

---

## Near-term priorities

1. **qcache-only startup** — stop loading the full FP32 model on cache hits.
2. **Generation controls** — temperature, top-p/top-k, repetition penalty, stop strings, seed.
3. **Correctness tools** — CPU/GPU logits diff, golden token tests, per-layer activation diff.
4. **Memory-safe quantization** — streaming safetensors → qcache, Q8_0 validation, mixed precision.
5. **llama.cpp parity for local UX** — templates, benchmarks, perplexity, OpenAI-compatible server.
6. **MoE differentiators** — Qwen MoE, expert hot/cold profiling, expert offload/prefetch.

---

## Documentation

- [Architecture](docs/ferrule_arch.md)
- [Roadmap and llama.cpp gap analysis](docs/ROADMAP.md)
- [Paper and system notes](docs/paper_reads.md)
