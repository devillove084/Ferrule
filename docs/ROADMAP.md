# Ferrule Roadmap

Ferrule's current milestone is **OLMoE-Instruct chat on GPU Q4_0** with router, top-k experts, and the expert loop preserved.

This roadmap is organized by priority, not dates. The immediate strategy is:

1. make the current MoE path reliable and easy to use
2. align the local UX with llama.cpp where it matters
3. add modern inference-system features in the right order
4. differentiate through sparse expert control, offload, rollout, and training state

---

## Current Capabilities

### Working now

- OLMoE safetensors loading
- OLMoE-Instruct chat-template shortcut
- CPU FP32 reference forward
- GPU Q4_0 quantized forward
- GPU Q8_0 GEMV kernels and offset expert GEMV path
- router → top-k experts → expert loop
- persistent single-session GPU KV cache
- llama.cpp-compatible Q4_0 block layout
- qcache for quantized per-layer weights
- `norm_topk_prob=false` router semantics
- interactive `ferrule chat`
- one-shot `run` and `gpu-run`
- CUDA probe/GEMV benchmark

### Main gaps

- qcache hits still load the full FP32 model first
- no sampling controls beyond greedy decode
- no formal CPU/GPU logits diff or golden-token regression suite
- no prompt/decode benchmark comparable to `llama-bench`
- no perplexity command
- no OpenAI-compatible server
- no batched prefill, continuous batching, paged KV, or prefix cache
- no FlashAttention-style kernel
- only OLMoE-style MoE is implemented end-to-end
- no LoRA/SFT/RL runtime yet

---

## Framework Alignment Snapshot

### llama.cpp capabilities to track

llama.cpp currently provides a broad local inference stack: GGUF model format, many model families, many quantization formats, CPU/GPU hybrid inference, multiple hardware backends, chat templates, sampling, grammar-constrained decoding, benchmarks, perplexity tools, server mode, embeddings/reranking, multimodal paths, and active model support.

Ferrule should not copy all of that immediately. The high-value subset for the next milestones is:

| Area | llama.cpp capability | Ferrule target |
|---|---|---|
| Local chat UX | robust CLI, templates, sampling | complete `chat` controls and template registry |
| Model cache | GGUF single-file metadata | qcache manifest + qcache-only startup |
| Quantization | broad Q/IQ/K quant suite | Q8 validation, mixed precision, K-quant/AWQ track |
| Quality checks | perplexity and eval tools | logits diff, golden tokens, perplexity |
| Performance | prompt/decode benchmark | `ferrule bench-infer` with pp/tg split |
| Serving | OpenAI-compatible server | minimal local HTTP server |
| Hybrid memory | CPU+GPU offload | expert-aware CPU/NVMe offload |
| Model coverage | many dense/MoE models | OLMoE → Qwen MoE → Mixtral/DeepSeek-style |

### Modern serving systems to track

| System line | Key idea | Ferrule implication |
|---|---|---|
| vLLM / PagedAttention | paged KV, continuous batching | needed for multi-session serving |
| SGLang | prefix/radix KV reuse, structured generation | useful after server and sampling exist |
| TensorRT-LLM / QServe | low-level kernel and quant co-design | later performance target |
| MLC-LLM / ExecuTorch | cross-device deployment | later backend portability target |
| PowerInfer / FlexGen / LLM in a flash | offload, hot/cold locality, flash/CPU scheduling | maps naturally to MoE expert offload |
| AWQ / TinyChat / KVQuant / KIVI | low-bit weight/KV quantization | informs Q4 quality and long-context memory work |

---

## P0 — Stabilize the Current MoE Runtime

Goal: make OLMoE-Instruct a dependable correctness baseline.

- [x] CPU FP32 OLMoE forward
- [x] GPU Q4_0 OLMoE forward
- [x] llama.cpp-compatible Q4_0 layout
- [x] OLMoE-Instruct chat smoke test
- [x] router `norm_topk_prob` semantics
- [x] gated token top-k debug logs via `FERRULE_DEBUG_TOPK`
- [ ] CPU/GPU logits comparison command
- [ ] per-layer activation diff tool
- [ ] golden token tests for short OLMoE-Instruct prompts
- [ ] CI-friendly tiny OLMoE fixture
- [ ] qcache metadata validation: magic, version, quant type, tensor count, shapes

Exit criteria:

- a short prompt can be validated automatically on CPU FP32 and GPU Q4_0
- qcache compatibility failures are explicit, not silent

---

## P1 — llama.cpp-Level Local Usability for OLMoE

Goal: make Ferrule comfortable as a local chat tool.

### Startup and loading

- [ ] qcache-only startup path
- [ ] avoid full CPU FP32 model load on qcache hits
- [ ] store/load small FP32 tensors required by GPU inference
- [ ] streaming safetensors → qcache writer
- [ ] shard/layer-by-layer quantization to stay under 32 GB RAM

### Generation UX

- [x] interactive `ferrule chat`
- [x] persistent KV cache across turns
- [x] EOS-aware output
- [ ] structured chat history instead of prompt fragments only
- [ ] tokenizer chat-template registry or `apply_chat_template` subset
- [ ] temperature
- [ ] top-k sampling
- [ ] top-p sampling
- [ ] repetition penalty
- [ ] stop strings
- [ ] seed control
- [ ] optional token/logprob output

### Local tools

- [ ] `bench-infer` with prompt-processing and token-generation metrics
- [ ] perplexity command over text files
- [ ] model/qcache inspection command
- [ ] generation config loading from `generation_config.json`

Exit criteria:

- a user can download, cache, chat, benchmark, and inspect OLMoE-Instruct without reading code

---

## P2 — Accuracy and Quantization Track

Goal: improve quality beyond naive Q4_0 while preserving edge usability.

- [x] Q8_0 quantizer
- [x] Q8_0 CUDA GEMV kernels
- [x] Q8_0 expert-offset GEMV dispatch
- [ ] Q8_0 end-to-end smoke test after memory-safe loading
- [ ] mixed precision policy: FP32/F16/Q8/Q4 by tensor class
- [ ] compare CPU FP32 vs GPU Q8_0 vs GPU Q4_0 logits
- [ ] evaluate Q4_0 divergence over longer generations
- [ ] investigate llama.cpp K-quants: Q4_K/Q5_K/Q6_K
- [ ] investigate AWQ-style activation-aware offline quantization
- [ ] investigate KV cache quantization after paged KV exists

Recommended order:

1. Q8_0 end-to-end validation
2. mixed precision policy
3. K-quant or AWQ-quality Q4 path
4. KV cache quantization

---

## P3 — Decode Engine and Serving

Goal: move from single-session CLI decode to a modern local serving runtime.

### Attention and KV

- [ ] multi-token prefill path
- [ ] FlashAttention-style exact attention kernel
- [ ] paged KV cache layout
- [ ] KV page allocator and free list
- [ ] prefix cache / radix cache design
- [ ] KV cache quantization prototype

### Scheduling

- [ ] reduce per-token kernel launch count
- [ ] CUDA graph capture or persistent decode schedule
- [ ] async host/device scheduling
- [ ] continuous batching
- [ ] speculative decoding

### Serving interface

- [ ] minimal HTTP server
- [ ] OpenAI-compatible `/v1/chat/completions`
- [ ] streaming responses
- [ ] request cancellation
- [ ] runtime metrics endpoint
- [ ] optional grammar/JSON constrained decoding

Exit criteria:

- Ferrule can serve multiple local chat sessions with predictable KV memory usage

---

## P4 — Larger MoE Models

Goal: run stronger MoE families without requiring huge CPU RAM.

Priority order:

1. `allenai/OLMoE-1B-7B-0924-Instruct`
   - current correctness baseline

2. `Qwen/Qwen1.5-MoE-A2.7B-Chat`
   - stronger chat ability
   - requires qwen2_moe loader
   - likely requires shared expert support
   - requires streaming quantization/qcache-only loading

3. `mistralai/Mixtral-8x7B-Instruct-v0.1`
   - standard top-2 MoE target
   - useful for expert offload and expert parallelism research
   - requires much stronger memory/offload path

4. DeepSeek-style MoE models
   - useful for long-term RL/MoE research
   - may require architecture-specific attention changes such as MLA

MoE-specific runtime work:

- [ ] expert activation counters per layer
- [ ] expert hot/cold profiling
- [ ] expert residency policy
- [ ] expert prefetch based on router history
- [ ] CPU/NVMe expert offload
- [ ] grouped expert execution
- [ ] fused multi-expert GEMV batching
- [ ] expert-parallel execution across devices
- [ ] router load-balance metrics

---

## P5 — Training and RL Runtime

Goal: evolve Ferrule workers into rollout workers, then connect them to fine-tuning and RL loops.

### LoRA/SFT

- [ ] LoRA adapter representation
- [ ] LoRA injection into attention and FFN projections
- [ ] adapter merge/unmerge path
- [ ] SFT data loader
- [ ] loss computation and backward graph prototype
- [ ] gradient accumulation
- [ ] optimizer state storage

### RL rollout

- [ ] sampling metadata and logprob capture
- [ ] rollout session format
- [ ] trajectory logging
- [ ] reward model/interface
- [ ] advantage computation
- [ ] PPO/GRPO experiment loop
- [ ] replay buffer / trajectory store

### MoE training-specific work

- [ ] router auxiliary loss
- [ ] expert load-balance logging
- [ ] expert gradient accumulation
- [ ] sparse expert optimizer state

---

## P6 — Elastic State Fabric

Goal: make model, qcache, KV, expert, rollout, and checkpoint state durable and movable.

- [ ] StateObject schema
- [ ] qcache manifest registry
- [ ] local StateAgent
- [ ] model version registry
- [ ] KV session binding
- [ ] checkpoint manifest registry
- [ ] transfer protocol for KV and qcache chunks
- [ ] session-first KV migration
- [ ] distributed rollout workers
- [ ] central trainer / policy update loop

---

## Success Metrics

| Metric | Current | Near target | Long target |
|---|---:|---:|---:|
| OLMoE-Instruct chat | works manually | golden tests | regression suite |
| cached startup | qcache exists, FP32 still loaded | qcache-only | remote qcache artifact |
| GPU Q4_0 decode | usable | measured diff | higher-quality quant |
| Q8_0 decode | kernels present | end-to-end validated | mixed precision policy |
| prompt/decode benchmark | none | pp/tg report | llama-bench style matrix |
| model family support | OLMoE | Qwen MoE | Mixtral/DeepSeek-style MoE |
| serving | CLI | local HTTP | paged KV + batching |
| training/RL | none | rollout logs | LoRA/SFT/RL loop |

---

## Immediate Next Steps

1. Implement qcache-only startup for OLMoE-Instruct.
2. Add CPU/GPU logits comparison and golden prompt tests.
3. Add sampling controls and chat-template registry.
4. Add prompt/decode benchmark and qcache inspection.
5. Validate Q8_0 end-to-end after qcache-only loading.
6. Start Qwen MoE loader design only after streaming quantization is in place.
