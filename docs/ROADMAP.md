# Ferrule Roadmap

_Last updated: 2026-07-04_

Ferrule is a Rust-native, state-aware LLM runtime for edge inference. The current
engineering center is **runtime graph backed Transformer execution** with
**DeepSeek-V4 Flash / DSpark** as the near-term pressure test and **OLMoE** as
the correctness golden model. DSV4 drives missing runtime capabilities but must
not become a model-specific runtime architecture — model-family names and raw
tensor names stay in `ferrule-model`; `ferrule-runtime` sees semantic roles,
graph ops, artifact groups, expert registries, KV state, and backend object
stores.

This roadmap merges the previous `ROADMAP.md` and `TEMP_INFERENCE_ROADMAP.md`
into a single source of truth, reflecting the current code state including the
`ferrule-storage` vocabulary crate and the `ExpertResidencyBackend` trait.

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
10. [Review checklist](#review-checklist)

---

## Current capabilities

### Working now

- **OLMoE correctness fixture**: safetensors loading, GPU Q4_0 inference
  through cuda-oxide kernels. Correct `norm_topk_prob` semantics as a
  regression fixture. CPU reference components (artifact decoders, router
  math, HC math, expert executor) anchor CUDA kernels.
- **DeepSeek V4 chat**: full 43-layer CUDA greedy generation + readline chat
  through `ferrule chat models/DeepSeek-V4-Flash-DSpark -q cuda
  --chat-template deepseek-v4`. Official DSV4 chat prompt wrapper works.
- **Runtime graph foundation**: `ferrule-graph` opaque IR, `GraphProgram`,
  `ExecutionBatch`, `ExternalBindingPlan`, `BackendObjectStore`,
  `ReferenceGraphBackend`, shape validation registry. Dense decoder graph +
  generic semantic Transformer graph paths.
- **Graph execution bridge**: coarse `transformer_state_init →
  transformer_layer → output_projection` path executes through
  `ReferenceGraphBackend` using typed artifact binders. Layer payloads cached
  per graph program/layer.
- **Model-family boundary**: `ferrule-model/src/families/` owns per-family HF
  tensor parsing. DSV4 names isolated in `families/deepseek_v4.rs`. Runtime
  consumes semantic artifact slices only.
- **DSV4 metadata + artifact materialization**: 72,317 tensors across 48 shards
  classified. Attention/HC/router/shared-expert/routed-expert tensor descriptors
  bind from local HF shards. Local-model tests skip cleanly when absent.
- **Expert streaming**: `ExpertStreamingPlanner` with slot-based GPU residency,
  LRU eviction, prefetch, and `commit_step`. `ExpertStreamingReader` reads
  bounded byte ranges from local safetensors. CPU packed-FP4 SwiGlU expert
  executor + correctness-first CUDA packed-FP4 expert executor.
- **Storage/residency vocabulary** (`ferrule-storage` crate): `StorageObjectId`
  (structured enum), `ObjectLocator`, `Placement`, `ObjectReplica`,
  `ReplicaHandleId`, `StorageCatalog` trait, `TransferEngine` trait +
  `MockTransferEngine`, `StorageResidencyPolicy` with frequency-aware eviction
  scoring. 67 tests passing.
- **`ExpertResidencyBackend` trait** (`ferrule-runtime`): unifies the
  duplicated load/evict loop from `routed_moe.rs` and `deepseek_v4.rs`.
  `apply_streaming_step()` provides one loop. `CpuExpertHandleStore` impl
  done. Adapters: `ExpertId → StorageObjectId`, `ExpertStorageTier →
  Placement`, `ExpertLoadSource → ObjectLocator`. 14 tests passing.
- **WeightPack**: mmap'd reader with manifest validation, streaming writer,
  zero-copy slices. WeightPack-only load path for OLMoE.
- **Quantization**: Q4_0/Q8_0 full path, mixed precision policy, FP4 E2M1 +
  E8M0 artifact decoders, FP8 E4M3FN decoders, KV quantization. K-quant/AWQ
  investigation done.
- **KV cache**: `KvCache` trait, contiguous per-session, radix prefix cache.
- **Scheduler**: continuous batching + preemption prototypes.
- **Structured decoding**: token mask API, program-like generation API.
- **CLI**: `info`, `run`, `gpu-run`, `chat`, `cuda`, `bench-infer`,
  `compare-logits`, `inspect-weightpack`, `server`, `deepseek-v4-probe`,
  `deepseek-v4-generate`, `perplexity`, `inspect-cache`.
- **Sampling**: temperature, top-k, top-p, min-p, repeat penalty, seed, stop
  strings, top-K logprobs. `generation_config.json` auto-loading. Chat template
  registry: OLMoE, ChatML, Llama3, Qwen, DeepSeek-V4, Plain.
- **Server**: minimal OpenAI-compatible `/v1/chat/completions` with SSE
  streaming.
- **Benchmarks**: `bench-infer` with JSON summary. `ferrule-bench` owns
  benchmark/report schemas.

### Main gaps

- DSV4 decode still has too many host-side `Vec<f32>` and synchronous
  artifact/operator boundaries (~0.3 tok/s estimated).
- Expert residency CUDA integration pending: `ExpertResidencyBackend` not yet
  implemented for `DeepSeekV4CudaOperatorCache`. Host-staged cache not built.
- No production batched selected-expert kernel.
- KV/compressed KV/indexer state not fully device-resident.
- Official DSV4 numeric parity incomplete.
- `ferrule-storage` vocabulary not yet wired into execution path (Phase 0 types
  only, no call site changes).
- No paged KV allocator, prefix cache integration with serving, or production
  continuous batching through the graph runtime.
- No DSpark speculation loop.
- No competitive PK harness.

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
9. **No async I/O on the correctness path** — keep synchronous bounded reader;
   replace backend with async/io_uring later without changing planning
   semantics.
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

### P0 — Correctness and observability gates

Goal: before optimizing DSV4, make every mismatch and bottleneck localizable.

- [x] Runtime graph validation and shape registry.
- [x] Generic semantic DSV4-capable graph builder without specialization.
- [x] Local DSV4 graph materialization smoke.
- [x] `ferrule-bench` owns benchmark/report/reference-smoke tooling.
- [x] DSV4/bench JSON summary schema.
- [x] Graph construction supports serializable summary for diffing.
- [ ] DSV4 first-token / first-N-token parity gate against official/reference.
- [ ] Layer-scoped debug dump: hidden, HC state, q/kv, attention output, router
      top-k, selected experts, logits top-k.
- [ ] Wire real backend telemetry into JSON schema: kernel launches/token,
      host↔device bytes/token, expert loads/token, peak VRAM/RSS.

### P1 — Graph execution bridge

Goal: `GraphProgram + BackendObjectStore` drives coarse `transformer_layer`
execution through existing generic components.

- [x] Coarse `transformer_state_init → transformer_layer → output_projection`.
- [x] Semantic graph externals materialize into artifact groups + expert
      registries + KV state.
- [x] `GraphLayerObjects` aggregation by `ArtifactGroupKind` + layer.
- [x] Typed binders: attention, HC, router, shared FFN from artifact groups.
- [x] Coarse `transformer_layer` lowering inside `ReferenceGraphBackend`.
- [x] Cache decoded payloads per graph program/layer.
- [x] Tiny/synthetic semantic layer through graph path.
- [ ] Local DSV4 layer 0 through the graph bridge when model is present.
- [ ] Debug dumps around the coarse layer boundary.

### P2 — Split `transformer_layer` into fine-grained semantic ops

Goal: expose performance-critical boundaries while staying model-family neutral.

- [ ] Split into phase ops: `layer_hc_pre`, `rms_norm`, `latent_attention`,
      `layer_hc_post`, `router_select`, `routed_moe`, `shared_ffn`,
      `residual_merge`.
- [x] Coarse graph attrs describe semantics (attention kind, KV shape, router
      kind, norm/HC eps, RoPE/YaRN, sliding/indexer metadata, SwiGLU limit).
- [ ] Attention/position/cache subgraph: `latent_q_project_a/b`,
      `latent_kv_project`, `rope_apply`, `window_kv_update`,
      `compressed_kv_update`, `indexer_score_topk`, `sparse_attention`,
      `attention_output_grouped_a/b`.
- [ ] Routed/shared MoE subgraph: `router_logits`, `hash_router_lookup`,
      `topk_select`, `route_weight_normalize`, `expert_registry_lookup`,
      `routed_expert_swiglu`, `shared_swiglu_ffn`, `moe_accumulate`.
- [ ] Shape inference for each semantic op.
- [ ] Validation rules rejecting raw HF tensor names in external keys.

### P3 — Device-resident DSV4 decode

Goal: remove host-mediated hot-path boundaries.

1. **Decode arena** — reusable device buffers for hidden, HC state, q/kv,
   attention workspace, router scores, expert outputs, logits/top-k.
2. **Device-resident KV state** — window KV, compressed KV, indexer KV,
   compressor state stay on device.
3. **Artifact linear dispatch** — generic APIs for BF16/F32/FP8/FP4 payloads;
   backend chooses CPU reference, CUDA dequant+matvec, or fused kernels.
4. **Expert residency** — persistent expert handles; selected experts reuse
   device handles; prefetch/evict policy visible in metrics.
5. **Output projection/top-k** — chunked/fused device-side lm_head top-k for
   greedy decode; copy back only token id + selected logit.

Exit: one-token DSV4 decode downloads only debug-gated tensors + final
token/logit; expert loads visible and amortizable.

### P4 — Storage and residency integration

Goal: wire `ferrule-storage` vocabulary into the execution path. See
`docs/storage-residency-architecture.md` for full design.

- [x] `ferrule-storage` crate with vocabulary types + traits (67 tests).
- [x] `ExpertResidencyBackend` trait + `apply_streaming_step()` + adapters
      (14 tests). `CpuExpertHandleStore` impl done.
- [x] Dead code `residency.rs` deleted.
- [ ] Implement `ExpertResidencyBackend` for `DeepSeekV4CudaOperatorCache`
      (CUDA path, `#[cfg(feature = "cuda")]`).
- [ ] Replace inline loops in `routed_moe.rs` and `deepseek_v4.rs` with
      `apply_streaming_step()`.
- [ ] Add `activation_count` to `ExpertState`; route eviction through
      `StorageResidencyPolicy` weights (recency + frequency).
- [ ] Add planner/cache consistency check + `planner_cache_inconsistencies`
      counter.
- [ ] Host-staged expert cache (`HostStagedExpertCache`) with per-tier
      hit/miss counters.
- [ ] Expert ID migration: `ExpertId` → `StorageObjectId` as handle store key.

### P5 — Kernel fusion and CUDA graph replay

Goal: optimize only after P0/P3/P4 make correctness and data movement visible.

- [ ] Count kernel launches per token/layer.
- [ ] Fuse low-rank attention projection + norm pieces.
- [ ] Fuse sparse attention gather/softmax/value accumulation.
- [ ] Batched selected-expert FP4/FP8 MoE kernel per layer.
- [ ] Grouped output projection fusion.
- [ ] Steady-state decode shape buckets.
- [ ] CUDA graph replay for stable decode buckets with debug fallback.

### P6 — Benchmarks and competitive parity

Goal: compare honestly against mainstream engines by scenario.

- [x] DSV4 JSON benchmark summary via `deepseek-v4-generate --json`.
- [ ] `bench-infer` DSV4 mode through graph bridge.
- [ ] Prompt/decode split comparable to `llama-bench`.
- [ ] Report: model artifact, recipe, precision, hardware, arch, prompt,
      context, batch, generated tokens, graph summary.
- [ ] PK matrix: local single-user (vs llama.cpp), CUDA optimized (vs TRT-LLM),
      serving (vs vLLM/SGLang, after scheduler), out-of-core (vs FlexGen).

### P7 — Serving runtime

Goal: multi-session serving through the graph runtime.

- [ ] Request/session/sequence lifecycle API.
- [ ] Paged KV allocator behind graph `KvState` handles.
- [ ] Chunked prefill.
- [ ] Continuous batching scheduler (prototype exists, not graph-backed).
- [ ] Prefix/radix KV reuse (radix cache exists, not integrated with serving).
- [ ] OpenAI-compatible `/v1/chat/completions` server (minimal version exists).
- [ ] Streaming responses, cancellation, metrics.
- [ ] Structured output masks / JSON constraints.

### P8 — DSpark / speculation

Goal: generic speculation around a correct target model.

- [ ] Represent MTP/draft artifacts as semantic `Speculation` bindings.
- [ ] Proposal model interface.
- [ ] Target verification interface.
- [ ] Acceptance/rejection/rollback state.
- [ ] Metrics: proposed, accepted, rejected, rollback, effective tok/s.
- [ ] Scheduler integration after serving state exists.

Rule: do not enable DSpark before base DSV4 correctness + device-resident
decode are stable.

---

## Implementation order

1. **Official parity gate** — tokenizer/template JSON parity, first-token /
   first-N-token golden tests, intermediate layer dumps.
2. **DSV4 benchmark contract** — JSON bench with load/prefill/decode split,
   tok/s, bytes moved, resident experts, peak GPU.
3. **Device-resident decode arena** — remove host `Vec<f32>` from per-operator
   boundaries.
4. **GPU KV/compressor/indexer state** — no host allocation/copy in decode.
5. **Expert residency + batched expert kernel** — `ExpertResidencyBackend`
   for CUDA, host-staged cache, one batched MoE kernel per layer.
6. **Attention/qkv fusion** — after residency, fuse projection/norm/rotary.
7. **GPU output-head/top-k** — greedy decode stays device-side.
8. **CUDA graph replay** — capture stable decode buckets.
9. **DSpark speculation** — only after base correctness + decode stable.
10. **Server PK** — only after scheduler/batching wired.

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
| CUDA performance | fused kernels, CUDA graphs, memory planners | many small kernels, correctness-first | decode arena, fusion, capture |
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
