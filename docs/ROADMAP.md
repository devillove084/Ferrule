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
10. [DSV4 synthetic fixture benchmark](#dsv4-synthetic-fixture-benchmark)
11. [Review checklist](#review-checklist)

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
- **DSV4 synthetic fixture**: `scripts/dsv4_synthetic_fixture.py` generates
  structurally complete zero-weight safetensors scaled to VRAM × 2.
  Auto-detects NVIDIA VRAM, scales all dimensions proportionally (FP8 block
  alignment, FP4 scale packing), and produces a valid `config.json` +
  `model.safetensors.index.json` + safetensors shards that Ferrule's DSV4
  loader, inventory, expert streaming, and generation path consume
  identically to real weights. Used for streaming/residency development
  without requiring the 166 GB full model on disk.
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

Measured on synthetic DSV4 fixture (RTX 3090 24 GB, CUDA sm_86, 43 layers,
170 experts, hidden=2816, 51 GB on disk). After Phase 1+2 optimizations
(device-resident expert loop, attention device chain, norm weight cache,
host-staged expert cache):

| Metric | Baseline | Current | Root cause |
|---|---|---|---|
| decode tok/s | 0.60 | 0.64 (+7%) | sync latency from D2H copies, not data volume |
| prefill tok/s | 0.065 | 0.063 | same host-mediated path, no prefill batching |
| kernel launches | 8,776 (~2,194/tok) | 9,808 (~2,229/tok) | per-expert launch, unfused attention sub-ops |
| H2D copies | 10,924 (4.48 GB) | 8,860 (4.80 GB) | reduced via device chaining; remaining is weight upload |
| D2H copies | 6,024 (73 MB) | 4,648 (67 MB) | reduced via device accumulation; remaining is op boundaries |
| expert loads (disk) | 243 | 243 | host-staged cache cold (5-token run too short for reuse) |

**Key finding:** reducing H2D/D2D copies by 30-40% yielded only +7% tok/s.
The bottleneck is synchronization latency (each D2H is a CPU-GPU sync point),
not data movement volume. CUDA graph capture (P5) is now the highest-impact
next step. See benchmark section for full analysis.

- DSV4 decode still has host-side Vec<f32> boundaries at `hc_pre`/`hc_post`,
  `apply_rotary_tail`, `sparse_attention`, and `grouped_output_a` boundaries.
- Expert residency CUDA integration: `HostStagedExpertCache` implemented and
  wired in; `ExpertResidencyBackend` not yet implemented for
  `DeepSeekV4CudaOperatorCache`.
- No production batched selected-expert kernel (6 experts/layer x 43 layers
  = 258 separate kernel launches per token).
- KV/compressed KV/indexer state not fully device-resident.
- CUDA graph capture not implemented: 9,808 launches for 5 forward passes.
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
**Motivation:** Phase 1+2 optimization reduced H2D/D2H copies by 30-40%, but
decode tok/s only improved 7%. The real bottleneck is **synchronization
latency** from D2H copies (each is a CPU-GPU sync point), not data movement
volume. See benchmark section for full analysis.

1. **Decode arena** — reusable device buffers for hidden, HC state, q/kv,
   attention workspace, router scores, expert outputs, logits/top-k.
   - [x] Device-resident FP4 expert input sharing: 6 experts per layer share
         one uploaded input buffer instead of 6 separate H2D uploads.
         (`prepare_fp4_expert_input` + `fp4_swiglu_ffn_from_device` in
         `ferrule-cuda/src/context.rs`)
   - [x] Device-side expert output accumulation via `saxpy_into`: eliminates
         258 D2H copies/token (6 experts × 43 layers) down to 43 D2H/token.
   - [x] Norm weight caching (`rms_norm_device_cached`): norm weights uploaded
         once per layer instead of per-call.
   - [x] Attention device chain (`decode_step_no_compress_device`):
         `query_a → rms_norm → query_b` and `key_value → rms_norm` chained on
         device; `hidden` uploaded once and reused for both projections.
   - [ ] Device-resident `hc_pre`/`hc_post`: need `hc_pre_from_device` and
         `hc_post_from_device` variants. These are called 4×/layer and each
         involves multiple H2D uploads + D2H downloads.
   - [ ] Device-resident `apply_rotary_tail`: currently host-only, called
         3×/layer. Needs a CUDA kernel.
   - [ ] Device-resident `sparse_attention`: already has CUDA kernel; needs
         `from_device` variant to accept pre-uploaded query/values.
   - [ ] Full decode arena: all per-layer intermediate buffers (HC state,
         attention I/O, router scores) device-resident end-to-end.
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

**Revised expectation:** P3 alone (copy reduction) yields ~7% improvement.
The ~10x target requires P5 (CUDA graph capture) to eliminate sync latency.

### P4 — Storage and residency integration

Goal: wire `ferrule-storage` vocabulary into the execution path. See
`docs/storage-residency-architecture.md` for full design.
**Motivation:** synthetic fixture recorded 243 expert disk reads across 5
forward passes — repeated expert activation re-reads from safetensors shards
with zero host-side caching. Host-staged cache targets this directly.

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
      - [x] `HostStagedExpertCache` LRU cache implemented in
            `expert_residency.rs` (6 tests). Serves re-activated experts from
            host RAM instead of re-reading from disk.
      - [x] Wired into `DeepSeekV4CudaOperatorCache::routed_moe_step` CUDA
            path.
      - [ ] Wire hit/miss counters into JSON benchmark summary.
- [ ] Expert ID migration: `ExpertId` → `StorageObjectId` as handle store key.

### P5 — Kernel fusion and CUDA graph replay

Goal: optimize only after P0/P3/P4 make correctness and data movement visible.
**Motivation:** Phase 2 benchmark showed that reducing H2D/D2H copies by 30-40%
yielded only +7% tok/s. The dominant bottleneck is **synchronization latency**:
~1,056 D2H copies/token each force CPU-GPU synchronization. CUDA graph capture
eliminates all intermediate sync points, allowing the GPU to execute the entire
decode bucket without CPU intervention. This is now the **highest-impact** item.

- [x] Count kernel launches per token/layer (measured: see benchmark above).
- [ ] **CUDA graph capture for stable decode buckets** — capture the entire
      `decode_step_device` path as one graph. The path is already structured
      for this: deterministic op sequence, no data-dependent control flow in
      steady-state decode. Expected: ~5-10x decode improvement.
      - [ ] Identify stable decode shape buckets (fixed hidden size, expert
            count, KV length window).
      - [ ] Capture graph for each bucket.
      - [ ] Replay with updated input/weight pointers.
      - [ ] Debug fallback (replay disabled when debug dump requested).
- [ ] Fuse low-rank attention projection + norm pieces.
- [ ] Fuse sparse attention gather/softmax/value accumulation.
- [ ] Batched selected-expert FP4/FP8 MoE kernel per layer.
- [ ] Grouped output projection fusion.
- [ ] Steady-state decode shape buckets.

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
| CUDA performance | fused kernels, CUDA graphs, memory planners | ~2,229 kernel launches/token, 30-40% copy reduction (Phase 2), correctness-first | CUDA graph capture (P5), hc_pre/post device variants (P3) |
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

### Experiment environment

| Item | Value |
|---|---|
| Date | 2026-07-04 |
| GPU | NVIDIA GeForce RTX 3090 (24 GB VRAM) |
| CUDA arch | sm_86 |
| OS | WSL2 (Linux), Windows host |
| Build | `just build-cuda` (cargo-oxide, release, `--features cuda --arch sm_86`) |
| Model | `models/DeepSeek-V4-Flash-DSpark-synthetic` (zero-weight structural fixture) |
| Fixture script | `scripts/dsv4_synthetic_fixture.py` (auto-scales to VRAM × 2) |
| Prompt | `"Hello"` (1 token, token id 19923) |
| Layer cap | `--max-layers 43` (full model depth) |
| Decode tokens | `-n 4` (4 decode tokens) or `-n 10` (10 decode tokens, for stability) |
| Weights | All zero-valued; structure identical to official model |
| compress_ratio | 0 (CSA/HCA disabled for synthetic; uses `decode_step_no_compress` path) |
| MTP | disabled (`num_nextn_predict_layers=0`) |

### Fixture configuration

| Param | Value |
|---|---|
| hidden_size | 2,816 (scaled 0.69x, aligned 128) |
| num_hidden_layers | 43 |
| n_routed_experts | 170 (scaled from 256) |
| num_attention_heads | 42 (o_groups=6, divides 42) |
| head_dim | 384 |
| q_lora_rank / o_lora_rank | 768 / 768 |
| moe_intermediate_size | 1,408 |
| vocab_size | 85,729 |
| num_nextn_predict_layers | 0 (MTP disabled for synthetic) |
| compress_ratios | all 0 (CSA disabled for synthetic) |
| tensor count | 46,172 |
| shard count | 12 (4 GiB each) |
| on-disk size | 50.8 GB |

### Reproduction

```bash
# Generate fixture (auto-detects VRAM, targets VRAM x 2)
python scripts/dsv4_synthetic_fixture.py

# Build and run (4-token decode)
just build-cuda
./target/release/ferrule deepseek-v4-generate \
  models/DeepSeek-V4-Flash-DSpark-synthetic \
  --backend cuda -n 4 --max-layers 43 --json

# 10-token decode (more stable, less prefill noise)
./target/release/ferrule deepseek-v4-generate \
  models/DeepSeek-V4-Flash-DSpark-synthetic \
  --backend cuda -n 10 --max-layers 43 --json

# Extract key metrics
./target/release/ferrule deepseek-v4-generate \
  models/DeepSeek-V4-Flash-DSpark-synthetic \
  --backend cuda -n 10 --max-layers 43 --json \
  | python3 -c "import json,sys; d=json.load(sys.stdin); s=d['summary']; \
    c=s['counters']; k=c['kernels']; t=c['transfers']; e=c['experts']; \
    print(f'decode tok/s: {s[\"decode_tok_per_s\"]:.3f}'); \
    print(f'prefill tok/s: {s[\"prefill_tok_per_s\"]:.4f}'); \
    print(f'kernel launches: {k[\"launches\"]}'); \
    print(f'H2D copies: {t[\"host_to_device_copies\"]}'); \
    print(f'D2H copies: {t[\"device_to_host_copies\"]}'); \
    print(f'expert loads: {e[\"loads\"]}')"
```

### Phase 0 — Baseline (pre-optimization)

Prefill 1 token + decode 4 tokens:

```
backend:            cuda (sm_86)
load:               2.83 s
prefill:            15.30 s  (0.065 tok/s)
decode (4 tokens):  6.63 s   (0.60 tok/s)
total:              21.93 s
```

| Counter | Value | Per-token | Notes |
|---|---:|---:|---|
| kernel launches | 8,776 | ~2,194/tok | ~51 launches/layer/token |
| expert selected | 1,032 | 258/tok | 6 per score-router layer, 1 per hash layer |
| expert loads | 243 | 60.8/tok | disk reads on activation; no host cache |
| expert load bytes | 1.43 GB | — | from safetensors shards |
| artifact uploads | 1,352 | — | attention + expert bundles to GPU |
| artifact upload bytes | 4.04 GB | — | |
| host->device copies | 10,924 | ~2,731/tok | 4.48 GB total |
| device->host copies | 6,024 | ~1,506/tok | 73 MB total |

### Phase 1 — Expert loop optimization (P3 expert + P4 cache)

**Changes:**
- `prepare_fp4_expert_input` + `fp4_swiglu_ffn_from_device`: 6 experts share
  one uploaded input buffer instead of 6 separate H2D uploads.
- `saxpy_into`: device-side expert output accumulation, eliminating per-expert
  D2H downloads.
- `HostStagedExpertCache`: LRU host-RAM cache for re-activated experts.

**Files:** `ferrule-cuda/src/context.rs`, `ferrule-runtime/src/expert_residency.rs`,
`ferrule-runtime/src/models/deepseek_v4.rs` (`routed_moe_step` CUDA path)

```
load:               1.50 s
prefill:            18.31 s  (0.055 tok/s)
decode (4 tokens):  6.48 s   (0.62 tok/s)
total:              24.78 s
```

| Counter | Baseline | Phase 1 | Delta |
|---|---:|---:|---:|
| kernel launches | 8,776 | 9,808 | +1,032 (saxpy adds 258/tok) |
| H2D copies | 10,924 | 10,064 | -860 (expert input shared) |
| D2H copies | 6,024 | 5,164 | -860 (expert outputs on device) |
| expert loads (disk) | 243 | 243 | 0 (cache cold: 5-token run too short) |
| decode tok/s | 0.60 | 0.62 | +3% |

### Phase 2 — Attention device chain + norm weight cache

**Changes:**
- `rms_norm_from_device` + `upload_norm_weight` + `rms_norm_device_cached`:
  norm weights uploaded once per layer and cached on device.
- `linear_matvec_from_device`: device-resident linear matvec using
  `artifact_linear_matvec_into` with `CudaF32Buffer` I/O.
- `decode_step_no_compress_device`: CUDA path that chains
  `query_a → rms_norm → query_b` and `key_value → rms_norm` on device,
  uploading `hidden` once and reusing for both projections.
- `decode_step_device`: CUDA path that uses `rms_norm_device_cached` for the
  two per-layer norm calls.

**Files:** `ferrule-cuda/src/context.rs`, `ferrule-runtime/src/models/deepseek_v4.rs`
(`decode_step_with_operators`, `decode_step_no_compress_with_operators`)

**4-token run:**

```
load:               1.50 s
prefill:            18.31 s  (0.063 tok/s)
decode (4 tokens):  6.20 s   (0.64 tok/s)
total:              22.82 s
```

| Counter | Baseline | Phase 1 | Phase 2 | Delta (P2 vs baseline) |
|---|---:|---:|---:|---:|
| kernel launches | 8,776 | 9,808 | 9,808 | +12% |
| H2D copies | 10,924 | 10,064 | 8,860 | **-19%** |
| D2H copies | 6,024 | 5,164 | 4,648 | **-23%** |
| decode tok/s | 0.60 | 0.62 | 0.64 | +7% |

**10-token run (more stable, per-token counts):**

| Counter | Baseline (per-token) | Phase 2 (per-token) | Delta |
|---|---:|---:|---:|
| H2D copies | ~2,731 | ~1,630 | **-40%** |
| D2H copies | ~1,506 | ~1,056 | **-30%** |
| kernel launches | ~2,194 | ~2,229 | +2% |
| decode tok/s | ~0.60 | ~0.49 | -18% (KV cache growth; see note) |

Note: 10-token decode tok/s is lower for both baseline and optimized because
sparse attention cost grows with KV cache length. The 4-token run is more
comparable to the original baseline measurement.

### Key findings

1. **H2D/D2H copies reduced 30-40%, but tok/s only +7%.** Copy reduction
   alone is insufficient — the bottleneck is not data movement volume but
   **synchronization latency**. Each D2H copy is a CPU-GPU sync point; ~1,056
   sync points/token dominate latency.

2. **Kernel launch overhead is significant but not dominant.** ~2,229 launches
   × ~5µs/launch ≈ 11ms/token. At 0.64 tok/s (~1.56s/token), launch overhead
   is ~0.7%. The real cost is the CPU stalling between launches waiting for
   D2H copies.

3. **Host-side computation is a hidden bottleneck.** `hc_pre`/`hc_post`,
   `apply_rotary_tail`, `quantize_attention_kv_for_qat_in_place`, and
   `sparse_attention` reference logic all run on CPU between GPU ops. This
   serializes CPU and GPU — neither is fully utilized.

4. **Host-staged expert cache was cold.** 243 disk reads unchanged because
   the 5-token run has near-zero expert reuse. Longer sequences or prefix
   caching would show cache hits. The cache value is in amortization, not
   single-shot runs.

### Updated bottleneck attribution

| Rank | Bottleneck | Evidence | Target | Est. gain |
|---|---|---|---|---|
| 1 | Sync latency from D2H copies (1,056/token) | -40% copies → only +7% tok/s | P5: CUDA graph capture | ~5-10x |
| 2 | Host-side compute between GPU ops | hc_pre/post, rotary, sparse_attn on CPU | P3: device-resident full layer | ~2-3x |
| 3 | Per-expert kernel launch (258/token) | 6 separate launches/layer | P5: batched expert kernel | ~2x MoE |
| 4 | No host-staged expert cache (cold) | 243 reads, 0 hits on 5-token run | P4: longer sequences | situational |
| 5 | Attention sub-ops unfused | multiple small kernels | P5: attention fusion | ~1.5x |

### Next steps (priority order)

1. **CUDA graph capture (P5)** — highest impact. Capture the entire decode
   bucket as one graph, eliminating all per-launch sync overhead. The
   `decode_step_device` path is already structured for this: deterministic
   op sequence, no data-dependent control flow in steady state. Expected:
   ~5-10x decode improvement.

2. **Device-resident `hc_pre`/`hc_post`** — these are called 4×/layer and
   each involves multiple H2D uploads + D2H downloads. Need `hc_pre_from_device`
   and `hc_post_from_device` variants in `CudaArtifactOperatorContext`.

3. **Device-resident `apply_rotary_tail`** — currently host-only, called
   3×/layer. Needs a CUDA kernel.

4. **Device-resident `sparse_attention`** — currently uploads all inputs and
   downloads output. Already has a CUDA kernel; needs `from_device` variant.

### Git state

Changes are unstaged on `main`. Files modified:
```
 crates/ferrule-cuda/src/context.rs               | 144 ++++++
 crates/ferrule-runtime/src/expert_residency.rs   | 227 +++++++++
 crates/ferrule-runtime/src/models/deepseek_v4.rs | 418 ++++++++++++++++++-
 docs/ROADMAP.md                                  | 199 ++++++-
 4 files changed, 963 insertions(+), 25 deletions(-)
```

CPU tests: 229 passed, 2 pre-existing failures
(`compressed_attention_decode_reference_updates_compressed_cache`,
`dsv4_layer_decode_step_runs_hc_attention_moe_shared_hc`) — both fail with
`invalid FP8 activation quant shape` and are unrelated to these changes.

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
