# Ferrule Architecture

Ferrule is a Rust-native, state-aware LLM runtime for edge inference. It targets
sparse MoE inference under memory pressure, with **OLMoE** as the correctness
golden model and **DeepSeek V4 Flash / DSpark** as the near-term pressure test.

The thesis: inference, rollout, quantization, cache management, and future
training should share one native systems foundation. Router decisions, selected
experts, quantized weights, KV cache, expert residency, and rollout state should
be explicit runtime objects that can be scheduled against real hardware.

---

## Table of Contents

1. [Crate layout](#crate-layout)
2. [Current runtime](#current-runtime)
3. [Runtime graph](#runtime-graph)
4. [Storage and residency](#storage-and-residency)
5. [Model-family boundary](#model-family-boundary)
6. [OLMoE forward pass](#olmoe-forward-pass)
7. [GPU runtime](#gpu-runtime)
8. [DeepSeek V4 execution](#deepseek-v4-execution)
9. [Quantization and WeightPack](#quantization-and-weightpack)
10. [KV cache](#kv-cache)
11. [Sampling and chat](#sampling-and-chat)
12. [Server](#server)
13. [Composable engine architecture](#composable-engine-architecture)
14. [Alignment targets](#alignment-targets)
15. [Future: Elastic State Fabric](#future-elastic-state-fabric)
16. [Future: training and RL](#future-training-and-rl)

---

![Ferrule architecture](assets/ferrule-current-architecture.svg)

## Crate layout

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ferrule-cli                                  │
│  chat · run · gpu-run · info · bench-infer · compare-logits ·       │
│  server · deepseek-v4-probe · deepseek-v4-generate · perplexity     │
└──────────┬──────────────────────────────────────────┬───────────────┘
           │                                          │
           ▼                                          ▼
┌─────────────────────┐  ┌──────────────────┐  ┌─────────────────────┐
│  ferrule-runtime    │  │  ferrule-cuda    │  │  ferrule-bench      │
│  runner · session   │  │  kernels · GPU   │  │  benchmarks · PK ·  │
│  sampler · scheduler│  │  forward pass    │  │  reference smokes   │
│  graph runtime      │  │  WeightPack      │  │  perplexity         │
│  expert streaming   │  │  artifact ops    │  │  JSON reports       │
│  expert residency   │  └──────────────────┘  └─────────────────────┘
│  routed MoE         │
│  KV cache · radix   │
└──┬──────┬───────────┘
   │      │
   ▼      ▼
┌──────┐ ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ferrule│ │ferrule-graph │  │  ferrule-model   │  │  ferrule-storage │
│-core │ │ opaque IR ·   │  │  OLMoE loader ·  │  │  vocabulary:     │
│errors│ │ GraphProgram │  │  DSV4 descriptor │  │  StorageObjectId │
│types │ │ shape registry│  │  families/ ·     │  │  Placement ·     │
│      │ │              │  │  tokenizer ·     │  │  ObjectLocator · │
│      │ │              │  │  policies        │  │  TransferEngine  │
└──────┘ └──────────────┘  └──────┬───────────┘  └──────────────────┘
                                   │
                                   ▼
                            ┌──────────────┐  ┌──────────────┐
                            │ ferrule-quant│  │ ferrule-gguf │
                            │ Q4/Q8/FP4/FP8│  │ safetensors  │
                            │ mixed policy │  │ GGUF reader  │
                            └──────────────┘  └──────────────┘
```

| Crate | Role |
|---|---|
| `ferrule-core` | shared errors, `QuantType`, observability |
| `ferrule-storage` | storage/residency vocabulary types + traits (no backend deps) |
| `ferrule-graph` | opaque graph IR, `GraphProgram`, shape registry |
| `ferrule-gguf` | GGUF and safetensors readers |
| `ferrule-quant` | Q4_0, Q8_0, FP4/FP8 artifact decoders, mixed precision policy |
| `ferrule-model` | OLMoE loader, DSV4 descriptor, model-family tensor policies, tokenizer |
| `ferrule-cuda` | cuda-oxide kernels, GPU forward pass, WeightPack reader, artifact operators |
| `ferrule-runtime` | runner, session, sampler, scheduler, graph runtime, expert streaming/residency, KV cache, structured decoding |
| `ferrule-bench` | benchmarks, PK harness, reference smokes, perplexity, JSON reports |
| `ferrule-cli` | CLI commands |

---

## Current runtime

```text
safetensors + tokenizer files
        ↓
ferrule-model family descriptor + tensor inventory
        ↓
Artifact-preserving runtime payloads
        ↓
DSV4: direct HF artifact streaming + mmap + managed expert handles
OLMoE: optional WeightPack cache / all-resident CUDA path
        ↓
DeepSeekV4ReferenceRunner / Engine-facing runner
        ↓
CUDA operator cache + session state + expert residency planner
        ↓
chat / one-shot generation / minimal server
```

Two execution paths coexist:

1. **OLMoE all-resident path**: `GpuOlmoeModel::build_from_cpu` quantizes and
   uploads all experts at startup as concatenated device buffers. It is the
   correctness fixture for router semantics and legacy CUDA kernels.

2. **DSV4 interactive streaming path**: `DeepSeekV4ReferenceRunner` owns session
   position and per-layer state. `DeepSeekV4CudaOperatorCache` owns CUDA operator
   state, uploaded linears, norm/rope caches, device KV buffers, expert handles,
   MoE workspaces, and counters. `ExpertStreamingPlanner` decides selected,
   prefetched, loaded, and evicted `(layer, expert)` bundles.

Current DSV4 chat is a **single-process resident runner**, not yet a production
serving engine:

- `ferrule chat ... -q cuda --chat-template deepseek-v4 --temp 0` starts the REPL
  immediately and loads CPU-side DSV4 artifacts on a background thread.
- CUDA context/operator cache is initialized on the main thread when the model is
  first used.
- Interactive prompt append uses `prefill_tokens_topk_interactive`: non-final
  prompt tokens update session/KV/MoE state without materializing final
  hidden/logits; the final prompt token materializes top-k for generation.
- This improves terminal latency, but it is still token-by-token device append,
  not true SGLang/vLLM-style CUDA chunked prefill.

---

## Runtime graph

The runtime graph is model-family neutral. Graph nodes describe semantics, not
families. Model-specific tensor names stay in `ferrule-model/src/families/`.

```
token_ids, positions
  → token_embedding
  → transformer_state_init
  → transformer_layer(layer=0..N, artifact groups, expert registry, kv state)
  → output_projection
  → logits_select
```

Current shape: coarse — one `transformer_layer` node per layer. The graph
execution bridge (`ReferenceGraphBackend`) lowers this to existing components:

```
GraphProgram + BackendObjectStore
  → GraphLayerObjects (aggregated by ArtifactGroupKind + layer)
  → typed artifact binders (attention, HC, router, shared FFN)
  → existing layer components (attention, HC, MoE, KV)
  → cached per graph program/layer
```

Next step: split `transformer_layer` into fine-grained semantic ops
(`layer_hc_pre`, `rms_norm`, `latent_attention`, `router_select`,
`routed_moe`, `shared_ffn`, `residual_merge`). See `docs/ROADMAP.md` P2.

Graph invariants:

- no CUDA buffers, KV pages, or WeightPack objects in graph nodes;
- no `deepseek` or raw HF tensor names in op names or external keys;
- unsupported policies fail with named semantic reasons.

---

## Storage and residency

Ferrule treats every loadable/resident runtime datum as a **storage object** with
identity, layout, locators, replicas, and policy. The vocabulary design is in
`docs/storage-residency-architecture.md`.

### Architecture

```text
StorageObjectId + descriptor + locators
        ↓
Residency policy scores objects by execute_now / predicted / recency / freq / cost
        ↓
Transfer/residency backend ensures placement
        ↓
Runtime handle store exposes executable objects to model code
```

### Current state

- `ferrule-storage` provides backend-neutral vocabulary: `StorageObjectId`,
  `ObjectLocator`, `Placement`, `ObjectReplica`, `ReplicaHandleId`,
  `StorageCatalog`, `TransferEngine`, and policy scoring types.
- `ExpertStreamingPlanner` is the active DSV4 strategy layer. It tracks per-layer
  expert state, recency, frequency, selected experts, predicted experts, loads,
  evictions, and committed residency.
- `HostStagedExpertCache` caches decoded expert bundles in host RAM and uses mmap
  shard reads underneath.
- CUDA DSV4 expert execution uses managed-memory FP4 expert buffers on GB10 by
  default. This avoids explicit H2D uploads for expert weights, but resident
  handles still represent `(layer, expert)` bundles and can exceed small-GPU
  budgets if left unbounded.
- Routing-aware hot prefetch is implemented: `--moe-prefetch-experts N` uses the
  per-layer observed hotset instead of naive low expert IDs.
- Bounded hotset residency is available via `--moe-hotset-experts N`; it is a
  memory-pressure tool. Short GB10 runs showed `N=16` increased churn and slowed
  throughput.

### Tier hierarchy

```text
Remote  ── future object store / LAN / RDMA
   ↓
Disk    ── HF safetensors / WeightPack / local cache log
   ↓
Host    ── mmap page cache + HostStagedExpertCache
   ↓
Managed ── CUDA managed expert buffers on unified-memory GB10
   ↓
Device  ── CUDA scratch, KV/cache, linears, output-head/top-k buffers
```

### Two residency policy levels

- `ferrule-model` `ResidencyPolicy` answers whether a model family allows
  streaming or requires all-resident execution.
- Runtime/storage policy answers what to keep, evict, prefetch, or pin under
  real budgets. For DSV4, this must be per `(layer, expert)`, not “256 experts”
  globally.

---

## Model-family boundary

```
crates/ferrule-model/src/families/
  common.rs       — conservative dense Transformer name policy
  deepseek_v4.rs  — DeepSeek V4 / Flash / DSpark HF tensor names + notes
```

The engine sees semantic roles and policy objects. A model family provides:

| Contract piece | Family provides | Engine provides |
|---|---|---|
| `ModelDescriptor` | architecture, dimensions, tensor inventory | format readers, validation |
| `ModelLayout` | layer graph, tensor classes, residual ordering | layer iteration, state ownership |
| `TensorBinding` | artifact names → semantic roles | loading, streaming, placement |
| `AttentionPolicy` | attention kind, projection roles, KV shape | scheduling, kernel dispatch |
| `ExpertPolicy` | routed/shared experts, routing semantics | batching, residency, execution |
| `QuantPolicy` | quant classes per role, fallback rules | validators, conversion, kernels |
| `ResidencyPolicy` | all-resident vs streaming | budget enforcement, eviction |
| `SpeculationPolicy` | draft/MTP attachment | propose/verify/rollback |

Rule: no model-specific names in generic crates. Add a policy/layout role, not a
field named after a specific model's tensor.

---

## OLMoE forward pass

Per token, each layer runs:

1. input RMSNorm
2. Q/K/V projections
3. Q/K head norms
4. RoPE
5. append K/V to KV cache
6. attention score, softmax, value combine
7. output projection and residual
8. FFN RMSNorm
9. router projection
10. top-k expert selection
11. expert gate/up/down loop
12. weighted expert accumulation and residual

Then final RMSNorm, `lm_head`, and token selection.

### Router correctness contract

- `norm_topk_prob=false`: softmax over **all experts**, select top-k, do not
  renormalize.
- `norm_topk_prob=true`: select top-k and renormalize.

This fixed the previous OLMoE-Instruct chat degeneration.

---

## GPU runtime

`ferrule-cuda` provides two families of CUDA code:

1. legacy OLMoE kernels over Q4/Q8/all-resident buffers;
2. artifact-preserving DSV4 operators over FP4/FP8/BF16/F32 payloads.

Persistent DSV4 CUDA state is owned by `DeepSeekV4CudaOperatorCache`:

- one CUDA context and stream through `CudaArtifactOperatorContext`;
- uploaded/cached artifact linears and norm weights;
- rope tables, sink buffers, grouped output-a cache, top-k buffer;
- device KV and combined window/compressed KV buffers where implemented;
- managed-memory FP4 expert handles in `experts: HashMap<ExpertId, ...>`;
- reusable routed MoE workspace and counters.

Key current kernels/operators:

| Kernel/operator | Purpose |
|---|---|
| `artifact_linear_*` | artifact-preserving FP4/FP8/BF16/F32 matvec/top-k |
| `fp4_e2m1_e8m0_quantize_f32_packed` | pack f32 activations into FP4 + E8M0 scales |
| `moe_gemm_dual_fp4_mxf4_batched` | FP4 mxf4 Tensor Core gate/up path |
| `moe_swiglu_fp4_packed_batched` | fused SwiGLU + route weight + FP4 pack |
| `moe_gemm_down_fp4_mxf4_batched` | FP4 mxf4 Tensor Core down projection |
| `moe_gemv_*_batched` | scalar/reduce FP4 fallback paths |
| `rms_norm_*`, `rope_tail_from_device` | device norm/rotary pieces |
| `sparse_attention_sink_from_device` | sparse attention over device query/value buffers |
| `artifact_linear_topk_from_device` | greedy output-head top-k without full logits download |

The FP4 Tensor Core path is correct and default-on (`FERRULE_CUDA_MOE_TC=0` forces
scalar fallback), but it is still under-utilized because current decode feeds
`batch_cols=1`. True GEMM utilization requires real prompt/speculative columns
and per-column routing.

---

## DeepSeek V4 execution

DSV4 is the current pressure test. It exercises hybrid attention, hyper-connection
state, compressed/sliding/indexer KV, hash + score/bias expert routing, shared
experts, routed FP4 experts, unified-memory residency, and DSpark/speculation
metadata.

### Current DSV4 path

```text
chat/generate token or prompt segment
  → tokenizer + DeepSeek-V4 chat template
  → runner session position + per-layer state
  → per layer:
       hc_pre → attn_norm → attention/KV append → hc_post
       hc_pre → ffn_norm → router → expert residency plan
       routed MoE + shared FFN → hc_post
  → final hc_head/output_norm only when hidden/logits are needed
  → device output-head top-k for greedy decode
```

Interactive chat adds a short-turn optimization:

```text
prompt tokens except last: feed_token() fast append, no final hidden/logits
last prompt token: decode_token_topk(), materialize top-k
new generated tokens: print token, then feed_token() if it must be committed
```

This reduces repeated final projection work, but it is still not true chunked
prefill. Every prompt token currently performs a full-layer append pass.

### DSV4 layer order

```text
hc_pre → attn_norm → attention → hc_post
  → hc_pre → ffn_norm → routed MoE + shared FFN → hc_post
```

### What works

- Full 43-layer CUDA greedy generation and readline chat.
- Official DSV4 chat template wrapper.
- Artifact inventory and binding for attention, HC, routers, shared experts, and
  routed experts from local HF shards.
- Hash-routed early layers and score/bias routed later layers.
- Managed-memory FP4 expert residency with mmap + host-staged cache.
- Device-resident hot decode chain for single-token greedy decode.
- FP4 `mxf4` Tensor Core MoE path with scalar fallback.
- Routing-aware expert hotset/prefetch counters.
- JSON decode benchmark counters and chat per-turn stats.

### Measured current behavior

- Best short JSON decode observed: **0.910 tok/s** for 8 decode tokens + 4 warmup
  with TC default and no prefetch.
- 16-token TC + hot prefetch observed: **0.864 tok/s**.
- Two-turn terminal pipe after interactive append:
  - REPL ready immediately; artifact load ~0.82s in background;
  - first turn `Hello`, `-n 1`: prefill 23.65s, decode/feed 1.28s;
  - second turn `Hi`, `-n 1`: prefill 5.88s, decode/feed 1.09s.

### What's missing

- Official DSV4 numeric parity and output-quality gate.
- True CUDA chunked/segment prefill over an existing prefix.
- Fully device-resident compressor/indexer state and top-k.
- Real FP4 GEMM utilization with `batch_cols > 1` and per-column routing.
- Resident worker abstraction shared by CLI chat and server.
- Paged KV, prefix/radix reuse, continuous batching, cancellation.
- DSpark proposal/verify/rollback loop.

---

## Quantization and WeightPack

### Q4_0

Ferrule Q4_0 matches llama.cpp `block_q4_0`:

```
block size: 32 values
storage:    16 bytes per block
byte j:     low nibble  = value j
byte j:     high nibble = value j + 16
scale:      d = max_value_at_absmax / -8
value:      (q - 8) * d
```

### Q8_0

Q8_0 quantizer and CUDA GEMV kernels exist, including expert-offset GEMV.

### Artifact formats (DSV4)

- FP4 E2M1 packed weights with E8M0 block scales (official DSV4 routed experts).
- FP8 E4M3FN activations with E8M0 block scales (DSV4 attention).
- BF16 norms, F32 sinks, I64 hash-router tables.

### WeightPack

WeightPack is Ferrule's native execution artifact:

```text
WeightPackManifest {
  format_version, quant_type, layout_version,
  model_config_hash, tensor_shapes, num_layers,
  per-layer packed bytes
}
```

- `WeightPackReader`: mmap'd zero-copy reader with manifest validation.
- `WeightPackSlice`: borrowed zero-copy view into packed layer data.
- `StreamingWeightPackWriter`: quantize one layer at a time, write
  incrementally, drop FP32 after quantization.

WeightPack-only load path exists for OLMoE. DSV4 uses direct artifact streaming
from HF safetensors (WeightPack conversion is a future option).

### Mixed precision policy

Per-tensor-class dtype policy: embeddings, attention projections, experts,
router, norms, `lm_head`, KV cache — each can have a different quantization
class. CLI: `--quant q4-mixed`.

---

## KV cache

- `KvCache` trait: `append`, `view`, `reset` with explicit `Handle`.
- Contiguous per-session KV (current production path).
- Radix prefix cache (module exists, not integrated with serving).
- Paged KV allocator (planned, not built).
- KV quantization: FP16 first, then KIVI/KVQuant-style low-bit.

---

## Sampling and chat

- llama.cpp-style sampling: temperature, top-k, top-p, min-p, repeat penalty,
  seed, stop strings, top-K logprobs.
- `generation_config.json` auto-loading with CLI > config > default precedence.
- Chat template registry: OLMoE, ChatML, Llama3, Qwen, DeepSeek-V4, Plain.
- Context controls: `--ctx-size`, `/reset`, `/stats`, `/experts`, `/ctx` in REPL.
- Token/logprob debugging: `--verbose-tokens`, `--logprobs <K>`.
- DSV4 greedy chat uses a top-k fast path and avoids full-vocab logits unless
  non-greedy/logprob settings require it.
- DSV4 chat now starts the REPL immediately and loads CPU-side artifacts in the
  background. The first prompt waits only if loading is not finished.
- DSV4 prompt append uses `prefill_tokens_topk_interactive` for short-turn latency:
  non-final prompt tokens update session state without final hidden/logits; the
  final prompt token produces top-k for generation.

The next architecture step is to move this CLI-local behavior into a reusable
resident `EngineWorker` so chat and server share the same SGLang-style execution
shape.

---

## Server

Minimal OpenAI-compatible server exists:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions` (one request at a time)
- SSE streaming (`stream: true`) with token text + final usage stats

Current limitation: server does not yet use the new DSV4 resident/lazy chat path
or a production worker loop. Production serving needs:

1. shared resident `EngineWorker` for chat and server;
2. request/session/sequence lifecycle;
3. paged KV and prefix/radix cache integration;
4. chunked prefill scheduler;
5. continuous batching for decode;
6. cancellation, metrics, and structured output masks.

---

## Composable engine architecture

The engine composes model-family policies into an `EnginePlan`:

```
ModelDescriptor → ModelLayout → ModelSupportContract → EnginePlan {
  ModelFamilyPolicy, AttentionPolicy, RouterPolicy, ExpertPolicy,
  QuantPolicy, KvPolicy, SchedulerPolicy, ParallelismPlan,
  ResidencyPolicy, SpeculationPolicy
}
```

Configuration switches (`engine`, `attention`, `kv`, `parallel`, `residency`,
`speculation`) map to typed policy objects. Model-specific assumptions must not
leak into scheduler/sampler/CLI.

| Policy | Near-term default | Future |
|---|---|---|
| `AttentionPolicy` | GQA RoPE | CSA/HCA/MLA, Flash prefill |
| `RouterPolicy` | dense top-k | bias/hash routing, EP |
| `ExpertPolicy` | routed experts | shared experts, batching, EP |
| `QuantPolicy` | Q4_0/Q8_0 WeightPack | GGUF K/IQ, FP4/FP8, mixed |
| `KvPolicy` | contiguous per-session | paged KV, prefix/radix |
| `SchedulerPolicy` | single request | continuous batching, preemption |
| `ResidencyPolicy` | all resident if fits | expert streaming, host-staged |
| `SpeculationPolicy` | none | MTP, DSpark, Eagle |

---

## Alignment targets

### Mainstream-engine parity that matters

| Capability | Ferrule today | Target |
|---|---|---|
| Chat CLI | immediate REPL + DSV4 lazy artifact load + interactive append | resident worker with async warmup and first-token metrics |
| Prompt prefill | token-by-token device append for CUDA chat | CUDA chunked prefill over existing prefix |
| Decode | ~0.9 tok/s short JSON DSV4; top-k device path | multi-token/speculative throughput + graph buckets |
| Model metadata | `info`, DSV4 descriptor, DSpark metadata parse | engine-plan inspection and policy dump |
| Quantization | Q4/Q8/FP4/FP8, mixed precision, artifact-preserving DSV4 | calibrated GGUF K/IQ + official quality gates |
| Benchmarks | JSON decode summaries + chat stats | interactive benchmark, first-token, PK matrix |
| Loading | safetensors + WeightPack; DSV4 background artifact load | resident worker warmup + artifact placement plan |
| Serving | minimal OpenAI-compatible server | SGLang/vLLM-style worker, paged KV, batching |
| Offload | expert streaming + managed memory + hotset prefetch | budgeted placement, async overlap, long-run policy |

### Differentiation

Ferrule's differentiation over dense local runtimes:

- **MoE-native**: explicit router/top-k/expert execution, expert hot/cold
  profiling, expert residency and prefetch.
- **Artifact-aware**: preserve official quantization formats (FP4/FP8) instead of
  always re-quantizing.
- **Interactive state-aware runtime**: session state, prompt append, KV, expert
  residency, and future speculation are explicit runtime objects.
- **Graph/runtime contracts**: semantic graph IR, typed artifact bindings, backend
  object store.
- **Storage/residency vocabulary**: `StorageObjectId`, `Placement`,
  `ObjectLocator`, `ObjectReplica` — one vocabulary for all loadable/resident
  state.
- **Rust-native**: no Python dependency, single binary, edge-deployable.

---

## Future: Elastic State Fabric

Elastic State Fabric is the future state layer for distributed rollout,
training, checkpointing, expert offload, and session migration. Not implemented.

Core objects:

| Object | Purpose |
|---|---|
| `StateObject` | atomic state item: WeightPack chunk, KV page, adapter, trajectory |
| `StateTablet` | ownership group for related state objects |
| `StateAgent` | worker-local cache, prefetch, eviction, transfer, verification |
| `ModelVersion` | runnable model identity: source, tokenizer, quant, layout, adapters |
| `SessionBinding` | `session_id → worker_id + kv_tablet_id + route_epoch` |

First implementation step: WeightPack manifest discipline. Useful locally today,
becomes the first Fabric-compatible artifact later.

---

## Future: training and RL

Ferrule should evolve in stages:

1. **Inference worker** — current single-GPU MoE decode.
2. **Rollout worker** — sampling, logprobs, trajectory records, model/adapter
   version tags.
3. **LoRA/SFT prototype** — adapter representation, injection/merge, minimal
   training loop.
4. **RL loop** — reward interface, advantage computation, PPO/GRPO-style driver.
5. **Distributed state** — WeightPack registry, trajectory store, checkpoint
   registry, session/expert placement.

A rollout trajectory should include: model version, adapter version, prompt
tokens, generated tokens, logprobs, rewards, terminal reason, sampling config,
and tool/environment events.
