# Ferrule

Agentic edge inference runtime. Rust-native. Multimodal-native.

## Architecture

See [ferrule_arch.md](docs/ferrule_arch.md) for the full design.
See [ROADMAP.md](docs/ROADMAP.md) for the development plan.

## Quick Start

```bash
# CPU build & benchmark
cargo build --release
./target/release/ferrule bench --hidden 2048 --layers 12

# GPU build (requires cuda-oxide toolchain)
cargo oxide build --features cuda --arch sm_86

# GPU probe & benchmark
./target/release/ferrule cuda

# GPU inference (OLMoE 1B, Q4_0 quantized)
./target/release/ferrule gpu-run models/OLMoE -p "Hello" -n 8 -q q4

# CPU inference
./target/release/ferrule run models/OLMoE -p "Hello" -n 16
```

## Crates

| Crate | Purpose |
|---|---|
| `ferrule-core` | Error types, QuantType enum |
| `ferrule-gguf` | GGUF + Safetensors format readers (mmap) |
| `ferrule-graph` | Persistent compute graph, CPU/CUDA backends |
| `ferrule-quant` | Quantization: Q4_0, Q2S, T1S block-wise quant + dequant |
| `ferrule-cuda` | CUDA kernels (gemv, rms_norm, attention, router) + forward pass |
| `ferrule-model` | Model architectures (OLMoE, future: Gemma MoE) |
| `ferrule-cli` | CLI: info, run, gpu-run, bench, cuda probe |

## Related

- [Elastic State Fabric design](docs/new_arch.md) — cloud control plane reference (future)
- [Paper survey](docs/paper_reads.md) — literature review on edge LLM inference
