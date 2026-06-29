# Ferrule Paper and System Notes

This document tracks the inference systems and papers that shape Ferrule's direction.

Ferrule's project vision:

> Build a Rust-native edge runtime where sparse MoE inference, quantized weights, KV/session state, expert placement, rollout trajectories, and future training state are first-class runtime objects.

The current milestone is OLMoE-Instruct chat on GPU Q4_0. The next question is how to move from a working prototype to a competitive local MoE runtime.

---

## 1. What Modern Local/Edge Inference Systems Are Optimizing

Recent inference systems converge on a few themes:

1. **Weight compression is not enough**
   - low-bit weights need kernels, packing, scaling policy, and accuracy validation
   - naive Q4 can run, but quality and speed require system-level co-design

2. **KV cache is now a central memory object**
   - long context and multi-session serving are KV-memory limited
   - paged KV and KV quantization are core runtime features, not optional extras

3. **Serving throughput is scheduling + memory management**
   - continuous batching, prefix reuse, and request scheduling are as important as matmul kernels

4. **Offload is becoming hardware-aware**
   - CPU, GPU, DRAM, NVMe, and flash each need explicit placement policies
   - MoE experts make this especially attractive because only a small subset is active per token

5. **Structured generation matters**
   - JSON/grammar constraints, tool calls, and agentic programs need runtime support

Ferrule should align with these trends while keeping its unique MoE focus.

---

## 2. Core Papers and Systems

| Area | Work | Key idea | Ferrule action |
|---|---|---|---|
| Local runtime baseline | [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF, broad quant formats, many backends, local CLI/server | match local UX for supported MoE models first |
| Paged KV | [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180) | virtual-memory-like KV blocks; 2-4x serving throughput | design paged KV after qcache-only startup |
| Attention kernels | [FlashAttention-2](https://arxiv.org/abs/2307.08691) | better GPU work partitioning for exact attention | replace simple attention path for prefill/long context |
| Hopper attention | [FlashAttention-3](https://arxiv.org/abs/2407.08608) | async/TMA/FP8 attention on H100 | long-term backend-specific target, not immediate for sm_86 |
| Weight quantization | [AWQ](https://arxiv.org/abs/2306.00978) | activation-aware 4-bit weight-only quantization | evaluate after Q8/qcache pipeline is stable |
| System quantization | [QServe](https://arxiv.org/abs/2405.04532) | W4A8KV4 with kernel/system co-design | use as guide for mixed weight+activation+KV policy |
| KV quantization | [KIVI](https://arxiv.org/abs/2402.02750) | asymmetric 2-bit KV; key per-channel, value per-token | revisit after paged KV exists |
| Long-context KV quant | [KVQuant](https://arxiv.org/abs/2401.18079) | sub-4-bit KV quantization for very long contexts | guide future long-context work |
| Offload | [FlexGen](https://arxiv.org/abs/2303.06865) | GPU/CPU/disk scheduling for limited memory | informs streaming quantization and offload planner |
| Consumer GPU locality | [PowerInfer](https://arxiv.org/abs/2312.12456) | hot/cold neuron locality and GPU/CPU hybrid execution | maps to MoE expert hot/cold residency |
| Flash/limited memory | [LLM in a flash](https://arxiv.org/abs/2312.11514) | load model chunks from flash using locality-aware patterns | informs NVMe/flash expert offload |
| Structured programs | [SGLang](https://arxiv.org/abs/2312.07104) | radix KV reuse and structured decoding runtime | later: prefix cache and JSON/grammar decoding |

---

## 3. Implications for Ferrule

### 3.1 qcache-only startup is the first real systems milestone

The current qcache accelerates cached startup by avoiding re-quantization, but Ferrule still loads the full FP32 model before using it.

That prevents larger MoE models and Q8 experiments on 32 GB RAM machines.

Required next step:

```text
config/tokenizer + qcache manifest
        ↓
load small FP32 tensors
        ↓
mmap quantized layer chunks
        ↓
upload to GPU
        ↓
run without full FP32 model residency
```

This is Ferrule's local version of the same principle behind FlexGen, PowerInfer, and LLM-in-a-flash: do not assume all state must be resident in the largest representation.

### 3.2 llama.cpp parity should be selective

llama.cpp has many mature capabilities. Ferrule should align with the subset that improves local MoE usability:

- chat templates
- sampling controls
- model/qcache inspection
- benchmark tooling
- perplexity/quality checks
- OpenAI-compatible local server
- hybrid CPU/GPU memory policies
- broader quantization formats

Ferrule should not immediately chase every backend or every model family. CUDA + sparse MoE is the current leverage point.

### 3.3 MoE experts are Ferrule's differentiator

Dense runtimes optimize layers and KV. Ferrule can optimize **expert state**:

- expert activation counters
- hot/cold expert profiles
- expert residency in VRAM
- expert prefetch from CPU/NVMe
- grouped expert execution
- expert-parallel execution across devices
- router statistics for load balancing and training

This is where Ferrule can become more than a Rust clone of an existing local runtime.

### 3.4 KV must become page-managed before serving

The current KV cache is a single-session contiguous buffer. That is fine for correctness, but it does not scale to multi-session serving.

Paged KV should come before continuous batching because it gives the scheduler a real memory allocator.

Suggested order:

1. single-session qcache-only startup
2. sampling and chat templates
3. prefill/decode benchmarks
4. paged KV layout
5. minimal HTTP server
6. continuous batching
7. prefix/radix cache

### 3.5 Quantization should become policy-driven

Current Q4_0 is a storage and kernel format. Future quantization needs a policy:

| Tensor class | Possible policy |
|---|---|
| embeddings | FP16/FP32 or row-wise quant later |
| attention projections | Q8/Q4/K-quant depending on accuracy |
| experts | Q4/AWQ-style, possibly expert-specific |
| router | FP32/FP16 because it is small and correctness-sensitive |
| norms | FP32/FP16 |
| `lm_head` | FP16/Q8 after validation |
| KV cache | paged FP16 first, then KIVI/KVQuant-style compression |

Q8_0 should be the next validation target because it gives a clean accuracy reference between CPU FP32 and GPU Q4_0.

---

## 4. Vision Statement

Ferrule's long-term goal is:

> A local-first sparse MoE runtime that can run, adapt, and train models under edge memory constraints by managing every important state object explicitly.

That means:

- **Inference**: fast local chat and serving on consumer GPUs
- **Memory**: qcache-only loading, streaming quantization, paged KV, expert offload
- **MoE**: router-aware profiling, expert cache policies, expert-parallel execution
- **Quality**: logits diff, perplexity, golden tests, better quantization formats
- **Training**: LoRA/SFT path sharing the same runtime state
- **RL**: rollout workers that record tokens, logprobs, rewards, and versions
- **Distributed state**: Elastic State Fabric for qcache, KV, adapters, trajectories, and checkpoints

The near-term project identity should be simple:

> Ferrule is the Rust-native edge runtime for sparse MoE models.

---

## 5. Reading Queue

High priority:

1. llama.cpp quantization and backend docs
2. vLLM PagedAttention
3. AWQ + TinyChat
4. QServe
5. KIVI / KVQuant
6. PowerInfer
7. FlexGen
8. SGLang

Model-family reading:

1. OLMoE architecture and `norm_topk_prob` behavior
2. Qwen MoE / qwen2_moe implementation details
3. Mixtral top-2 routing and expert layout
4. DeepSeek-style MoE and attention variants

Training/RL reading:

1. LoRA / QLoRA adapter placement
2. GRPO/PPO rollout data requirements
3. distributed checkpoint and model-version registries
4. expert load-balancing losses and router auxiliary losses
