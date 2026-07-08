# Ferrule Roadmap

_Last updated: 2026-07-07_

Ferrule is a Rust-native, state-aware LLM runtime for edge inference. The current
engineering center is **interactive, resident, CUDA-backed DeepSeek-V4 Flash /
DSpark execution** on GB10-class unified-memory machines, while **OLMoE** remains
the correctness golden model.

The working target is no longer only an offline `decode_tok/s` number. Ferrule
must optimize the user-visible path:

1. `./target/release/ferrule chat ...` reaches the prompt quickly;
2. every user turn appends new prompt tokens without redoing old prefix work;
3. the first response token and the full response arrive with minimal latency;
4. benchmark JSON still reports load / prefill / decode counters for regression
   tracking.

DSV4 drives missing runtime capabilities but must not become a model-specific
runtime architecture. Model-family names and raw HF tensor names stay in
`ferrule-model`; `ferrule-runtime` sees semantic roles, graph ops, artifact
groups, expert registries, KV state, residency policy, and backend object stores.

---

## Table of Contents

1. [Current capabilities](#current-capabilities)
2. [Architecture principles](#architecture-principles)
3. [Composable engine architecture](#composable-engine-architecture)
4. [Priority roadmap](#priority-roadmap)
5. [Implementation order](#implementation-order)
6. [What not to do](#what-not-to-do)
7. [Gap against mainstream engines](#gap-against-mainstream-engines)
8. [Model bring-up contract](#model-bring-up-contract)
9. [DSV4 artifact facts](#dsv4-artifact-facts)
10. [DSV4 synthetic fixture benchmark](#dsv4-synthetic-fixture-benchmark)
11. [GB10 hardware analysis and measured state](#gb10-hardware-analysis-and-measured-state)
12. [Review checklist](#review-checklist)

---

## Current capabilities

### Working now

- **OLMoE correctness fixture**: safetensors loading, GPU Q4_0 inference through
  cuda-oxide kernels, and router correctness around `norm_topk_prob`. CPU
  reference components remain the anchor for artifact decoders, router math, HC
  math, expert execution, and CUDA tests.
- **DeepSeek V4 / DSpark metadata and local artifacts**:
  - local model: `models/DeepSeek-V4-Flash-DSpark`;
  - `hidden_size=4096`, `moe_intermediate_size=2048`, `num_hidden_layers=43`;
  - `n_routed_experts=256`, `num_experts_per_tok=6`, routed expert dtype `fp4`;
  - DSpark metadata is parsed: `dspark_block_size=5`, noise token, target layers
    `[40, 41, 42]`, Markov rank `256`.
- **Full 43-layer DSV4 CUDA greedy path**:
  - `deepseek-v4-generate --backend cuda --json` reports load / prefill / decode
    timing and CUDA counters;
  - `ferrule chat ... -q cuda --chat-template deepseek-v4 --temp 0` runs the
    DSV4 greedy top-k fast path;
  - `lm_head` top-k can stay device-side for greedy decode.
- **Interactive DSV4 chat improvements (2026-07-07)**:
  - chat REPL becomes ready immediately while DSV4 artifacts load on a background
    thread;
  - CUDA context/operator cache is still created on the main thread;
  - prompt append uses `prefill_tokens_topk_interactive`: all prompt tokens except
    the last update KV/MoE/session state without materializing final hidden/logits;
    the last prompt token materializes top-k for generation;
  - `feed_token` now advances session state without unnecessary `hc_head`, output
    norm, or `lm_head` work.
- **DSV4 CUDA execution state**:
  - device-resident decode chain exists for the hot single-token path;
  - attention/KV append, norm weight cache, GPU output-head/top-k, batched routed
    MoE workspace, managed-memory FP4 expert handles, and mmap shard reads are in
    the working path;
  - `FERRULE_CUDA_MOE_TC=1` (default) enables the FP4 `mxf4` Tensor Core path;
    `FERRULE_CUDA_MOE_TC=0` forces scalar FP4 fallback for A/B.
- **Blackwell FP4 Tensor Core smoke**:
  - local `fp4_mxf4_smoke` proves
    `mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale...ue8m0` on
    `sm_121a` with correct full-tile scale selection;
  - the integrated MoE path uses packed FP4 activations + E8M0 scales and has a
    fused `SwiGLU -> packed FP4` stage, eliminating the previous hidden-pack pass.
- **Expert residency / storage**:
  - `ExpertStreamingPlanner` tracks per-layer hot expert frequency and can
    prefetch routing-aware hot experts instead of naive low expert IDs;
  - bounded hotset residency (`--moe-hotset-experts`) exists for constrained
    machines but is not the fastest short-run default;
  - `HostStagedExpertCache`, mmap shard cache, and managed expert allocations are
    wired into CUDA DSV4.
- **Runtime graph foundation**: `ferrule-graph` opaque IR, `GraphProgram`,
  `ExecutionBatch`, `ExternalBindingPlan`, `BackendObjectStore`,
  `ReferenceGraphBackend`, and shape validation registry.
- **Model-family boundary**: `ferrule-model/src/families/` owns per-family HF
  tensor parsing. DSV4 raw tensor names are isolated in `families/deepseek_v4.rs`.
- **Storage/residency vocabulary** (`ferrule-storage`): `StorageObjectId`,
  `ObjectLocator`, `Placement`, `ObjectReplica`, `ReplicaHandleId`,
  `StorageCatalog`, `TransferEngine`, and frequency-aware scoring.
- **Quantization / artifacts**: Q4_0/Q8_0 full path, mixed precision policy,
  FP4 E2M1 + E8M0 artifact decoders, FP8 E4M3FN decoders, KV quantization, and
  WeightPack reader/writer for OLMoE.
- **CLI / serving surface**: `info`, `run`, `gpu-run`, `chat`, `cuda`,
  `bench-infer`, `compare-logits`, `inspect-weightpack`, `server`,
  `deepseek-v4-probe`, `deepseek-v4-generate`, `perplexity`, `inspect-cache`.

### Main gaps

Current measurements are on GB10 / `sm_121a` with the local DSV4 Flash DSpark
artifact unless noted. The most important gap is **end-to-end interactive
latency**, not just standalone decode throughput.

| Area | Current known state | Gap |
|---|---|---|
| Chat startup | REPL prints immediately; artifact load observed at ~0.82s in a two-turn pipe run | CUDA/operator warmup still happens on first real prompt |
| First user turn | `Hello`, `-n 1`: prefill **23.65s**, decode/feed **1.28s**, pos=6 | prompt append is still token-by-token device decode, not true chunked prefill |
| Second user turn | `Hi`, `-n 1`: prefill **5.88s**, decode/feed **1.09s**, pos=11 | every new prompt token still costs a full-layer device pass |
| JSON decode throughput | best current short run observed: **0.910 tok/s** (`8 decode`, `4 warmup`, TC default, no prefetch) | far below 5 tok/s milestone and far below theoretical memory ceiling |
| FP4 Tensor Core | correct and default-on; small short-run gains, MoE timing improves TC substage | still `batch_cols=1`; not real DSpark/GEMM utilization |
| Prompt/prefill | interactive path avoids final hidden/logits for non-final prompt tokens | no CUDA segment prefill / paged prefix scheduler |
| Compressor/indexer | still has host-side pieces and D2H boundaries | blocks clean graph capture and efficient chunked prefill |
| Serving | minimal OpenAI-compatible server exists | no resident worker loop, paged KV, continuous batching, prefix cache integration |
| DSpark | metadata parsed | no real speculative block decode or verification loop |

Important measurement corrections:

- Short `run:` wall-clock timings are misleading; use JSON summary for decode
  throughput and explicit chat `stats>` lines for interactive latency.
- `FERRULE_CUDA_MOE_TIMING=1` inserts stream syncs and is profiling-only.
- Bounded hotset residency can reduce resident handles, but on short runs it can
  increase churn and slow down throughput.
- Duplicating the same hidden vector to fake `dspark_block_size=5` would be wrong;
  real GEMM utilization needs real per-column hidden states and route semantics.

---

## Architecture principles

1. **Keep CPU reference components alive** — artifact decoders, router math,
   HC math, and expert executors are correctness anchors. The legacy full-model
   CPU forward path has been removed; reference is component-level, not a runner.
2. **Model families are policies, not runners** — plug into layout/attention/
   router/quant/residency policies; never fork a runner per family.
3. **No privileged target-model path** — generic crates expose semantic roles
   and policy traits, not DeepSeek-named fields.
4. **Measure before optimizing** — add benchmarks and counters before kernel
   rewrites.
5. **Treat WeightPack as a runtime artifact** — Ferrule's local/edge deployment
   unit with manifest, chunks, checksums, and future placement metadata.
6. **Preserve MoE semantics** — router logits, top-k, expert weights, and
   normalization are correctness-critical and must stay explicit.
7. **Compose behavior through typed policies** — `EnginePlan`/config surface,
   not `if model_name` branches.
8. **Make unsupported execution explicit** — `info` may inspect metadata;
   `run/chat/server` must fail with a named missing-policy error.
9. **Async I/O is allowed for expert loading** — the correctness path uses
   synchronous bounded reader; expert weight loading uses tokio + `pread` for
   concurrent disk reads. Planning semantics remain synchronous and unchanged.
10. **PK against real engines with reproducible commands** — every perf claim
    needs prompt, context, quant, batch, hardware, and quality note.

---

## Composable engine architecture

A new model bring-up provides descriptor parsing, tensor layout, and policy
implementations; the runtime composes them into an `EnginePlan`:

```
ModelDescriptor → ModelLayout → ModelSupportContract → EnginePlan {
  ModelFamilyPolicy, AttentionPolicy, RouterPolicy, ExpertPolicy,
  QuantPolicy, KvPolicy, SchedulerPolicy, ParallelismPlan,
  ResidencyPolicy, SpeculationPolicy
}
```

| Policy | Responsibility | Near-term default | Future switches |
|---|---|---|---|
| `ModelFamilyPolicy` | map metadata/tensor names → semantic layer components | OLMoE fixture, DSV4 descriptor | family plugins without runner forks |
| `AttentionPolicy` | MHA/GQA/DeepSeek attention, RoPE, cache layout | GQA RoPE | CSA/HCA/MLA, Flash prefill |
| `RouterPolicy` | logits, bias, hash routing, top-k normalization | dense top-k | bias/hash routing, EP routing |
| `ExpertPolicy` | dense MLP, routed/shared experts, activation | routed experts | shared experts, expert batching, EP |
| `QuantPolicy` | tensor quant class, dequant/matvec, conversion | Q4_0/Q8_0 WeightPack | GGUF K/IQ, FP4/FP8, mixed |
| `KvPolicy` | contiguous/paged/quantized KV ownership | contiguous per-session | paged KV, prefix/radix reuse |
| `SchedulerPolicy` | prefill/decode admission, chunking, batching | single request | continuous batching, preemption |
| `ParallelismPlan` | DP/TP/EP/SP/CP/PP placement | single process/GPU | all combinations |
| `ResidencyPolicy` (model-level) | all-GPU vs streaming allowed | all resident if it fits | expert/layer streaming |
| `StorageResidencyPolicy` (runtime-level) | per-tier budgets, eviction, prefetch | — | frequency-aware, host-staged |
| `SpeculationPolicy` | none/MTP/draft-model proposal + acceptance | none | MTP, DSpark, Eagle |

**Two residency policy levels** (see `docs/storage-residency-architecture.md`):

- `ferrule-model`'s `ResidencyPolicy` (`streaming_allowed`,
  `all_resident_required`) — model-level: does this model fit in GPU memory?
- `ferrule-storage`'s `StorageResidencyPolicy` (`Budgets`, `EvictionWeights`,
  `prefetch_window`) — runtime-level: how much to keep on each tier, what to
  evict. Active only when `streaming_allowed = true`.

---

## Priority roadmap

The priority order is now based on the user-visible interactive path:

```text
startup → prompt append/prefill → first token → steady decode → serving/multi-session
```

### P0 — Correctness and observability gates

- [x] JSON benchmark summary for `deepseek-v4-generate --json` with decode-only
      counters after prefill/warmup baseline subtraction.
- [x] MoE timing counters behind `FERRULE_CUDA_MOE_TIMING=1`.
- [x] DSV4 chat `stats>` line reports per-turn prefill/decode wall time.
- [x] Add a dedicated interactive benchmark command (`bench-interactive`) that
      feeds multiple turns and reports time-to-REPL, artifact load,
      first-token latency, per-turn prefill, per-turn decode, generated tok/s,
      and resident/cache counters.
- [x] Add golden interactive traces (`configs/golden/`) with token IDs and stop
      behavior for a fixed two-turn prompt under greedy decoding;
      `bench-interactive --golden <path>` compares live output against golden.

### P1 — Resident interactive engine

Goal: match SGLang-style execution at the architecture level before chasing more
micro-kernels: a long-lived engine owns model artifacts, CUDA context, KV/cache,
residency state, and request/session state.

- [x] DSV4 chat REPL starts immediately while artifacts load in the background.
- [x] CUDA context stays on the main thread; background loader only materializes
      CPU-side model artifacts.
- [x] `feed_token` appends session state without materializing final hidden/logits.
- [x] `prefill_tokens_topk_interactive` uses device decode append for prompt tokens
      and materializes top-k only for the last prompt token.
- [ ] Replace the CLI-local lazy runner with a reusable `EngineWorker` abstraction
      used by chat and server.
- [ ] Engine worker API: `load_async`, `append_prompt`, `decode_next`, `reset`,
      `stats`, cancellation, and structured counters.
- [ ] Persistent warmup policy: after artifact load, optionally warm common CUDA
      kernels/linears without blocking the REPL.

### P2 — True CUDA chunked/segment prefill

Goal: eliminate per-prompt-token full-layer device passes. This is the current
largest interactive latency gap.

- [ ] Implement CUDA segment prefill for DSV4 prompt chunks:
      `tokens_per_chunk > 1`, existing session prefix, per-layer processing.
- [ ] Attention segment append should update window KV, compressed KV, and indexer
      state without resetting compressor state.
- [ ] Keep prompt chunk hidden states on device; avoid host `Vec<f32>` boundaries.
- [ ] Only materialize top-k/logits for the final prompt token unless explicitly
      requested.
- [ ] Add chunk-size sweep benchmark for chat prompts (1, 2, 4, 8, 16 tokens).

### P3 — Device-resident compressor/indexer path

Goal: remove remaining host boundaries that block efficient chunked prefill and
CUDA graph capture.

- [x] Device-resident decode chain exists for the hot single-token path.
- [x] Norm weights, rope tables, output-head top-k, and MoE workspace are cached.
- [ ] Move compressor softmax/state update to CUDA.
- [ ] Move indexer top-k/selection to CUDA.
- [ ] Keep combined window/compressed values device-resident across attention.
- [ ] Reconcile CPU reference and CUDA path with parity tests at layer boundaries.

### P4 — Expert residency and FP4 MoE execution

Goal: reduce MoE cost without breaking correctness or faking DSpark batching.

- [x] Managed-memory FP4 expert handles default-on for GB10 unified memory.
- [x] Host mmap cache + `HostStagedExpertCache` are wired.
- [x] Routing-aware hot expert prediction replaces naive low-ID prefetch.
- [x] Batched selected-expert MoE workspace avoids per-call scratch allocation.
- [x] FP4 `mxf4` Tensor Core path is integrated and default-on, with scalar
      fallback via `FERRULE_CUDA_MOE_TC=0`.
- [x] Fused Tensor-Core SwiGLU output pack removes the separate hidden-pack stage.
- [ ] Generalize MoE kernels for real `batch_cols > 1` with per-column routing.
- [ ] Convert DSpark/speculative blocks into real hidden columns before using GEMM
      as the main speed lever.
- [ ] Add correctness A/B tests against scalar MoE for integrated gate/up/down.

### P5 — CUDA graph and launch/fusion work

Graph capture is not the first lever while per-token work is >1s, but it becomes
important after P2/P3/P4 reduce compute and host boundaries.

- [x] cuda-oxide fork exposes graph APIs and FP4 support; Ferrule also has raw
      graph handle scaffolding.
- [ ] Pre-allocate all decode/prompt buffers needed by stable buckets.
- [ ] Capture stable single-token decode buckets after compressor/indexer is
      device-resident.
- [ ] Capture prompt chunk buckets after CUDA segment prefill exists.
- [ ] Fuse low-rank attention projection + norm + rotary where profiler confirms
      launch overhead matters.

### P6 — Benchmarks and competitive parity

- [x] ROADMAP-compatible JSON decode summaries exist.
- [x] Current two-turn chat pipe measurement is recorded below.
- [ ] Add `bench-interactive` or `deepseek-v4-chat-bench` with reproducible
      multi-turn prompts and first-token latency.
- [ ] PK matrix by scenario: local CLI vs llama.cpp, CUDA optimized vs TRT-LLM,
      serving vs vLLM/SGLang, out-of-core vs FlexGen.
- [ ] Report model artifact, precision, hardware, arch, prompt, context, batch,
      generated tokens, graph summary, and quality/parity note.

### P7 — Serving runtime

- [x] Minimal OpenAI-compatible server exists.
- [ ] Move DSV4 chat/server onto the same resident `EngineWorker`.
- [ ] Request/session/sequence lifecycle API.
- [ ] Paged KV allocator behind graph `KvState` handles.
- [ ] Prefix/radix KV reuse integrated with sessions.
- [ ] Continuous batching scheduler for decode.
- [ ] Chunked prefill scheduler for prompt bursts.
- [ ] Streaming responses, cancellation, metrics, structured output masks.

### P8 — DSpark / speculation

- [x] DSpark metadata is parsed from config.
- [ ] Represent DSpark/MTP artifacts as semantic `Speculation` bindings.
- [ ] Proposal model interface.
- [ ] Target verification interface.
- [ ] Acceptance/rejection/rollback state.
- [ ] Metrics: proposed, accepted, rejected, rollback, effective tok/s.
- [ ] Scheduler integration after resident engine state exists.

Rule: do not fake `dspark_block_size=5` by duplicating hidden vectors. Enable
DSpark only when the runtime has real speculative hidden states, per-column route
semantics, and rollback-safe KV/residency state.

---

## Implementation order

1. **Interactive benchmark contract** — add a repeatable two-/multi-turn chat
   benchmark with time-to-REPL, load, prompt append, first token, full response,
   and resident/cache counters. Do this before more kernel work.
2. **Resident engine worker** — factor DSV4 chat/server onto one long-lived worker
   that owns CUDA context, model artifacts, KV/session state, and residency state.
3. **True CUDA chunked prefill** — replace per-prompt-token full-layer append with
   device-resident segment prefill over existing prefix state. This is the biggest
   current user-visible latency lever.
4. **Device compressor/indexer** — remove remaining D2H/CPU boundaries needed by
   chunked prefill and graph capture.
5. **Real MoE GEMM / DSpark columns** — extend FP4 mxf4 kernels to true
   `batch_cols > 1` with per-column routing; only then use DSpark block size as a
   throughput lever.
6. **Correctness parity gates** — keep tokenizer/template, first-token,
   first-N-token, and layer-boundary checks alongside CUDA changes.
7. **CUDA graph replay** — capture stable decode/prompt buckets after P2/P3 remove
   host work and P4 makes kernel-launch overhead meaningful.
8. **Paged KV + prefix cache serving** — integrate radix/paged KV with request
   lifecycle and continuous batching.
9. **Competitive PK** — compare against llama.cpp/TRT-LLM/SGLang/vLLM only with
   reproducible prompts and quality notes.

---

## What not to do

- Do not implement continuous batching before request/session state and KV
  abstraction exist.
- Do not implement radix cache before prefix cache and shareable KV pages exist.
- Do not delete CPU FP32 inference — it is the correctness anchor.
- Do not add broad model-family support before loader/runtime boundaries are
  cleaner.
- Do not rewrite all CUDA kernels before `bench-infer` and profiling exist.
- Do not add a new quantization format without `compare-logits`, perplexity, or
  golden-token harness.
- Do not make WeightPack format changes without manifest/version checks.
- Do not add DeepSeek-specific fields to generic scheduler/sampler/session/
  executor — add a policy/layout role instead.
- Do not claim DSV4 execution until attention, router, experts, auxiliary
  tensors, and quant classes are all accounted for.
- Do not claim DSpark support until generic target/draft speculation with
  acceptance/rejection/rollback/metrics is implemented.
- Do not compare Ferrule against llama.cpp/vLLM/SGLang without recording model,
  quant, prompt, context, batch, hardware, and command.
- Do not block on async I/O — keep the synchronous bounded reader as the
  correctness path.
- Do not enable DSpark before base DSV4 correctness and device-resident decode
  are stable.

---

## Gap against mainstream engines

| Area | Mainstream engines | Ferrule now | Ferrule gap |
|---|---|---|---|
| Model format | GGUF / engine plans / mature packaging | HF safetensors, WeightPack, semantic graph bindings | WeightPack-only startup, manifests, GGUF compatibility |
| Model coverage | many dense/MoE families | OLMoE executable; DSV4 metadata + artifact graph | full DSV4 graph execution, broader adapters |
| Attention kernels | FlashAttention/FlashMLA, paged KV | correctness-first reference + CUDA kernels | fused DSV4 latent/sparse attention, device-resident KV |
| MoE execution | batched/fused expert kernels, EP | expert streaming planner, artifact bundles, CPU/CUDA correctness paths | production batched kernel, residency/prefetch, GPU handle reuse |
| Storage/residency | ad-hoc per-engine | `ferrule-storage` vocabulary + `ExpertResidencyBackend` trait | wire vocabulary into execution, host-staged cache |
| Quantization | GPTQ/AWQ/FP8/INT4/K/IQ, calibration | Q4/Q8 + FP4/FP8 artifact decoders, mixed policy | calibration, GGUF K/IQ execution, quality gates |
| Scheduler | continuous batching, chunked prefill, preemption | scheduler prototypes, not graph-backed | paged KV + request lifecycle + graph batch lowering |
| CUDA performance | fused kernels, CUDA graphs, memory planners, tensor core | 1,849 launches/token (Phase 4), scalar FP4 GEMV, +53% from device chain | **Tensor core FP4 GEMM** (P5a), expert pre-load, CUDA graph (P5b) |
| Correctness | evals, perplexity, golden suites, parity | unit/local smokes, compare tools | DSV4 official parity, long regressions |
| Serving UX | OpenAI API, streaming, metrics | minimal server, CLI tools | full server after graph runtime stabilizes |
| Distributed | TP/PP/EP, multi-node | none | design should not block, not near-term |

Ferrule's intended differentiation: **explicit runtime state + artifact-aware
MoE residency + Rust-native graph/runtime contracts**.

---

## Model bring-up contract

Every new model family satisfies the same contract. DSV4 is the first hard
target; OLMoE is the correctness fixture.

| Contract piece | Model family provides | Engine provides |
|---|---|---|
| `ModelDescriptor` | architecture, dimensions, tokenizer metadata, tensor inventory | format readers, metadata validation |
| `ModelLayout` | layer graph, required/optional tensor classes, residual/norm ordering | layer iteration, state ownership |
| `TensorBinding` | artifact tensor names → semantic tensor roles | loading, mmap/streaming, placement |
| `AttentionPolicy` | attention kind, projection roles, position encoding, KV shape | attention step scheduling, KV ownership |
| `MlpPolicy` / `ExpertPolicy` | dense MLP or routed/shared experts, activation semantics | expert batching, residency, execution |
| `QuantPolicy` | supported quant classes per tensor role, fallback rules | validators, conversion, kernels |
| `TokenizerPolicy` | tokenizer files, chat templates, special tokens | prompt/session lifecycle, sampler |
| `ValidationPolicy` | reference commands, golden tokens/logits, tolerances | compare tools, regression storage |
| `SpeculationPolicy` | optional draft/MTP attachment + acceptance rules | proposal/verify loop, rollback, metrics |

Source layout rule:

```
crates/ferrule-model/src/families/
  common.rs       # conservative dense Transformer name policy
  deepseek_v4.rs  # DeepSeek V4 / Flash / DSpark HF+GGUF names
```

When adding a family, create `families/<family>.rs` that maps source artifact
names into generic `TensorClass` / `TensorRole` values. Do not add artifact
tensor names to generic `tensor_policy`, runtime runners, KV state, or CUDA
kernels.

---

## DSV4 artifact facts

From local shard headers of `deepseek-ai/DeepSeek-V4-Flash-DSpark`:

| Group | Bytes | GiB | Implication |
|---|---:|---:|---|
| Routed expert gate | 52,479,131,648 | 48.875 | dominates memory |
| Routed expert up | 52,479,131,648 | 48.875 | dominates memory |
| Routed expert down | 52,479,131,648 | 48.875 | dominates memory |
| Attention tensors | 5,721,447,936 | 5.329 | preserve artifact dtype |
| Shared experts | 1,157,698,560 | 1.078 | conservative Q8 |
| Embedding | 1,059,061,760 | 0.986 | conservative Q8 |
| Output head | 1,059,061,760 | 0.986 | conservative Q8 |
| DSpark/MTP | 198,786,204 | 0.185 | keep input dtype until verifier |

Quantization decision: **quality-first default** — preserve official artifact
FP4 for routed experts, implement streaming/residency. Do not re-quantize to
4-bit as the single-node fit strategy. Design streaming before extra
quantization.

---

## DSV4 synthetic fixture benchmark

This section is retained as historical context only. The current optimization
priority is based on the real local model and interactive chat measurements in
the GB10 section below.

### Fixture purpose

`scripts/dsv4_synthetic_fixture.py` creates structurally valid DSV4-like
safetensors without requiring the full 166 GB real model. It is useful for
loader, inventory, graph binding, expert streaming, and residency development.

The fixture preserves:

- DSV4-style top-level config and tokenizer boundary;
- layer counts and semantic tensor roles;
- FP8 block-aligned attention artifacts;
- FP4 routed expert packing and E8M0 scale shapes;
- shard/index structure compatible with the real loader.

### What the old fixture measurements proved

Historical synthetic/fixture work helped validate these architecture changes:

1. DSV4 artifact inventory and typed binders can materialize complete layer
   payloads.
2. Expert streaming can load bounded safetensors byte ranges and install resident
   handles.
3. Batched selected-expert MoE and device-resident attention pieces reduce obvious
   host/device boundaries.
4. JSON counters are required; raw wall-clock command timing was too noisy.

### What is now stale

The old fixture-era bottleneck attribution, next-step list, and git-state notes
were removed because they conflict with the real-model measurements. In
particular:

- CUDA graph capture is no longer the first priority while each prompt/decode
  token still costs seconds.
- The current highest-priority user-visible bottleneck is per-prompt-token
  full-layer append; this needs CUDA chunked prefill and a resident engine worker.
- MoE Tensor Core work remains important, but the current `batch_cols=1` path is
  not enough for the 5 tok/s target.

---

## GB10 hardware analysis and measured state

### Hardware and model facts

| Item | Value |
|---|---|
| GPU | NVIDIA GB10 / Blackwell, `sm_121a` build target |
| Memory | 128 GB unified LPDDR5X class system memory |
| Model | `models/DeepSeek-V4-Flash-DSpark` |
| DSV4 shape | 43 layers, hidden 4096, MoE intermediate 2048 |
| Experts | 256 routed experts/layer, top-6 experts/token |
| Expert dtype | FP4 E2M1 packed weights + E8M0 block scales |
| DSpark metadata | block size 5, target layers `[40, 41, 42]`, Markov rank 256 |

The theoretical ceiling remains orders of magnitude above current performance;
current work is dominated by runtime architecture and under-utilized kernels, not
raw memory bandwidth alone.

### Measurement methodology

Use JSON decode summaries for throughput and chat `stats>` lines for interactive
latency. Avoid comparing raw shell `run:` time across code paths because it mixes
artifact load, prefill, decode, cold CUDA setup, expert residency, and stdout.

Canonical decode command:

```bash
./target/release/ferrule deepseek-v4-generate \
  models/DeepSeek-V4-Flash-DSpark \
  --prompt "Hello" --backend cuda --max-tokens 8 \
  --warmup-tokens 4 --output-head-chunk-rows 4096 \
  --json --chat
```

Profiling-only MoE timing:

```bash
FERRULE_CUDA_MOE_TIMING=1 ./target/release/ferrule deepseek-v4-generate ... --json
```

`FERRULE_CUDA_MOE_TIMING=1` inserts stream synchronizations and should not be
used as a throughput result.

### Current JSON decode measurements (2026-07-07)

All rows use `--prompt Hello --chat --warmup-tokens 4 --output-head-chunk-rows
4096` unless noted. `TC` means the integrated FP4 `mxf4` path. `Scalar` means
`FERRULE_CUDA_MOE_TC=0`.

| Output file | Tokens | MoE path | Prefetch | Hotset budget | decode tok/s | decode s | Expert loads | Evictions | Resident experts | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `target/dsv4-tc-default-8w4.json` | 8 | TC | 0 | 0 | **0.910** | 8.789 | 365 | 0 | — | best current short JSON run |
| `target/dsv4-hotset-default-8w4.json` | 8 | Scalar | 0 | 0 | 0.865 | 9.250 | 388 | 0 | 1628 | managed planner default, no eviction |
| `target/dsv4-hotprefetch8-8w4.json` | 8 | Scalar | 8 | 0 | 0.897 | 8.918 | 389 | 44 | 1585 | routing-aware prefetch |
| `target/dsv4-tc-hotprefetch8-8w4.json` | 8 | TC | 8 | 0 | 0.902 | 8.870 | 375 | 40 | — | TC + hot prefetch |
| `target/dsv4-hotset16-prefetch8-8w4.json` | 8 | Scalar | 8 | 16 | 0.827 | 9.672 | 605 | 605 | 688 | bounded hotset slowed short run |
| `target/dsv4-hotset-default-16w4.json` | 16 | Scalar | 0 | 0 | 0.740 | 13.515 | ~48.4/tok | 0 | 1728 | noisy longer scalar baseline |
| `target/dsv4-hotprefetch8-16w4.json` | 16 | Scalar | 8 | 0 | 0.817 | 12.233 | ~48.6/tok | 80 | 1642 | hot prefetch helps longer run |
| `target/dsv4-tc-hotprefetch8-16w4.json` | 16 | TC | 8 | 0 | 0.864 | 11.571 | ~47.0/tok | 77 | — | TC + prefetch best measured 16-token run |
| `target/dsv4-fused-scalar-8w4.json` | 8 | Scalar | 0 | legacy | 0.835 | 9.578 | 388 | 43 | 1578 | before managed planner no-evict alignment |
| `target/dsv4-fused-tc-8w4.json` | 8 | TC | 0 | legacy | 0.725 | 11.033 | 364 | 40 | 1554 | old TC path before residency/default fixes |

Interpretation:

- TC is now default because current A/B is no longer a clear regression and is
  required for future real GEMM/DSpark utilization.
- The integrated TC path at `batch_cols=1` is not enough for 5 tok/s. It needs
  real multi-column hidden states, not duplicated data.
- Bounded hotset is a memory-pressure tool, not a short-prompt speed default.
- Routing-aware prefetch can help 8/16-token decode but does not change the main
  latency class because prompt/decode still run full-layer token steps.

### MoE profiling measurements

With `FERRULE_CUDA_MOE_TIMING=1` on a 2-token run, timing syncs make absolute
`decode_tok/s` meaningless, but stage attribution is useful:

| Path | decode tok/s under timing | MoE total | Hidden pack | Notes |
|---|---:|---:|---:|---|
| TC fused | 0.219 | 0.110s | 0.000s | `SwiGLU -> packed FP4` fused; hidden pack removed |
| Scalar | 0.216 | 0.155s | n/a | scalar FP4 fallback |

Conclusion: the fused TC MoE substage improves MoE time, but end-to-end latency
is still dominated by prompt/decode architecture, attention/compressor/indexer
host work, and under-utilized single-column MMA.

### Interactive chat measurements (2026-07-07)

Command:

```bash
printf 'Hello\nHi\n/exit\n' | \
  ./target/release/ferrule chat models/DeepSeek-V4-Flash-DSpark \
  -q cuda -n 1 --chat-template deepseek-v4 --temp 0
```

| State | First turn prefill | First turn decode/feed | Second turn prefill | Second turn decode/feed | Notes |
|---|---:|---:|---:|---:|---|
| Before interactive append | 29.285s | 9.341s | 10.487s | 1.322s | non-final prompt/generated tokens materialized hidden/logits |
| After interactive append | **23.648s** | **1.278s** | **5.878s** | **1.089s** | REPL ready immediately; artifact load ~0.82s background |

This is the most important current user-facing result. The remaining multi-second
per-turn prefill shows that Ferrule still lacks true CUDA chunked prefill over an
existing prefix. SGLang/vLLM-style serving engines win here because prompt chunks,
KV pages, scheduler state, and decode workers are resident and batched.

### Updated bottleneck attribution

| Rank | Bottleneck | Current impact | Target fix |
|---|---|---|---|
| 1 | Per-prompt-token full-layer append | multi-second prefill for every user turn | CUDA chunked/segment prefill over existing prefix |
| 2 | Compressor/indexer host work | prevents efficient chunked prefill and graph capture | device compressor/indexer kernels and state |
| 3 | Under-utilized FP4 TC MoE (`batch_cols=1`) | TC is correct but not a real GEMM throughput unlock | real DSpark/speculative columns with per-column routing |
| 4 | Expert residency churn under constrained budgets | bounded hotset can slow short runs if too small | routing-aware budget/pinning, longer-run policy tuning |
| 5 | Kernel launch fragmentation | important after token work drops below current ~1s class | CUDA graph stable buckets after P2/P3 |
| 6 | Startup/warmup | REPL ready fixed; first prompt still pays CUDA/operator warmup | resident engine worker + optional async warmup |

### Performance projection

| Phase | Expected user-visible effect | Status |
|---|---|---|
| Background artifact load | faster time-to-REPL | done |
| Fast `feed_token` / interactive append | lower per-turn decode/feed overhead | done |
| CUDA chunked prefill | largest remaining prompt latency drop | next priority |
| Device compressor/indexer | enables chunked prefill + graph capture | next priority |
| Real FP4 GEMM with DSpark columns | path to multi-token throughput improvement | blocked on real speculative state |
| CUDA graph replay | reduces launch overhead after host boundaries are gone | later |
| Paged KV + continuous batching | serving parity with SGLang/vLLM class engines | later |

### Next-step priority list (2026-07-07)

| Priority | Task | Why now |
|---|---|---|
| **P1** | Add interactive benchmark harness | Prevent optimizing the wrong metric again |
| **P2** | Build resident `EngineWorker` shared by chat/server | SGLang-style architecture starts here |
| **P3** | CUDA chunked prefill over existing prefix | Biggest measured per-turn latency gap |
| **P4** | Device compressor/indexer | Required by chunked prefill and graph capture |
| **P5** | Real `batch_cols>1` FP4 MoE with per-column routes | Converts TC correctness into GEMM throughput |
| **P6** | DSpark proposal/verify/rollback state | Needed before using `dspark_block_size=5` honestly |
| **P7** | CUDA graph buckets | Useful after host/token work is reduced |
| **P8** | Paged KV + continuous batching | Serving/PK stage |

---

## Review checklist

For every patch:

- **Behavior**: does it change inference output? Is the change tested?
- **Correctness**: do CPU reference components still pass? Does compare-logits
  regress?
- **Generality**: does it add model-specific names to generic crates?
- **Performance**: does it add host sync to a device path? Does it add kernel
  launches?
- **State ownership**: does it leak state into the wrong crate?
- **Compatibility**: does it break existing CLI commands or WeightPack files?
- **Safety**: does it add `unsafe` without justification?
