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

```
safetensors + tokenizer files
        ↓
ferrule-model loader (OLMoE) or family descriptor (DSV4)
        ↓
CPU FP32 tensors (OLMoE) or artifact payloads (DSV4)
        ↓
Q4_0 / Q8_0 quantization (OLMoE) or artifact-preserving FP4 (DSV4)
        ↓
WeightPack cache (optional, OLMoE) or direct artifact streaming (DSV4)
        ↓
CUDA device buffers
        ↓
ferrule-runtime session + sampler + graph runtime
        ↓
chat / one-shot generation / server
```

Two execution paths coexist:

1. **OLMoE legacy path**: `GpuOlmoeModel::build_from_cpu` quantizes and uploads
   all experts at startup as a concatenated `DeviceBuffer`. All-resident, no
   eviction, no streaming. Used when the model fits in VRAM.

2. **DSV4 streaming path**: `DeepSeekV4CudaOperatorCache` holds per-expert
   `CudaFp4ExpertHandles` in a `HashMap`. `ExpertStreamingPlanner` decides
   load/evict per layer per token. Synchronous disk read + H2D upload on miss.
   Used when the model does not fit in VRAM.

Both paths are valid. The storage abstraction accommodates both — all-resident
mode bypasses eviction; streaming mode uses `ExpertResidencyBackend`.

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

Ferrule treats every loadable/resident runtime datum as a **storage object**
with identity, layout, locators, replicas, and policy. The design is in
`docs/storage-residency-architecture.md`.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STORAGE OBJECT                              │
│  StorageObjectId (enum)  ·  Descriptor  ·  ObjectLocator(s)         │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                    catalog maps ID → descriptor + locators
                                   │
         ┌─────────────────────────┴───────────────────────────┐
         ▼                                                   ▼
┌─────────────────────────┐              ┌────────────────────────────┐
│    ResidencyManager     │   miss →     │      TransferEngine        │
│  ObjectReplica[]        │  transfer    │  ensure() / prefetch()      │
│  placement · state ·    │              │  poll()                     │
│  generation · handleId  │              │  backends: file/mmap/H2D/   │
│  budgets (bytes+slots)  │              │  (future) io_uring/RDMA     │
└─────────────────────────┘              └────────────────────────────┘
         ▲                                                   ▲
         │  policy ranks objects, requests residency          │
         ▼                                                   │
┌─────────────────────────────────────────────────────────────────────┐
│  StorageResidencyPolicy { budgets, retain_hot, eviction_weights }   │
│  ResidencyScore { execute_now, predicted, recency, freq, cost }     │
└─────────────────────────────────────────────────────────────────────┘
```

### Current state

- `ferrule-storage` crate: vocabulary types + traits, zero backend deps, 67
  tests.
- `ExpertResidencyBackend` trait in `ferrule-runtime`: unifies the duplicated
  load/evict loop. `CpuExpertHandleStore` impl done. 14 tests.
- Two active systems: `ExpertStreamingPlanner` (strategy) +
  `DeepSeekV4CudaOperatorCache` (execution). Not merged — they are different
  layers. The trait unifies the execution loop, not the systems.
- Dead code `residency.rs` deleted.

### Tier hierarchy

```
Remote  ── (Phase 5: RDMA / object store / LAN cache)
   │
   ▼
Disk    ── safetensors / WeightPack / local cache log
   │
   ▼
Host    ── staged bytes (pinned or unpinned)     [Phase 2: not built yet]
   │
   ▼
Device  ── CUDA handles / device buffers
```

### Two residency policy levels

- `ferrule-model` `ResidencyPolicy` (`streaming_allowed`,
  `all_resident_required`) — model-level: does this model fit?
- `ferrule-storage` `StorageResidencyPolicy` (`Budgets`, `EvictionWeights`,
  `prefetch_window`) — runtime-level: what to keep, what to evict. Active only
  when `streaming_allowed = true`.

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

`GpuOlmoeModel` owns one CUDA context, one stream, persistent device buffers,
and reusable scratch buffers.

Persistent state:

- embedding table and `lm_head` in FP32
- norm weights and router weights in FP32
- quantized attention projections
- quantized expert projections (concatenated `ex_gate_packed` / `ex_down_packed`)
- RoPE tables
- per-layer K/V cache
- hidden, attention, router, expert, and logits scratch buffers

Key kernels:

| Kernel | Purpose |
|---|---|
| `embed_lookup` | token embedding lookup |
| `gemv_f32` | router and `lm_head` projections |
| `gemv_q4`, `gemv_q4_off` | Q4_0 GEMV and expert-offset GEMV |
| `gemv_q8`, `gemv_q8_off` | Q8_0 GEMV and expert-offset GEMV |
| `gemv_dual_q4_off` | fused Q4_0 expert gate/up GEMV |
| `gemv_triple_q4` | fused Q/K/V path when dimensions match |
| `rms_norm_fused` | RMSNorm reduction and apply |
| `rope` | rotary position embedding |
| `attn_scores` | GQA-aware attention scores |
| `attn_combine_softmax` | softmax and V accumulation |
| `router_topk` | all-expert softmax and top-k selection |
| `silu_mul` | SiLU(gate) × up |
| `saxpy` | weighted accumulation and residual updates |
| `artifact_fp4_swiglu_ffn_matvec` | DSV4 packed FP4 expert SwiGLU |

---

## DeepSeek V4 execution

DSV4 is the near-term pressure test. It exercises: hybrid attention (CSA/HCA/
MLA), sparse expert routing (hash + score/bias), artifact FP4 experts, HC
(hyper-connection) state, compressed/sliding/indexer KV, and out-of-core
streaming pressure.

### Current DSV4 path

```
router selects experts (per layer)
  → ExpertStreamingPlanner.plan_layer_step()
       produces loads[] + evictions[]
  → DeepSeekV4CudaOperatorCache.routed_moe_step():
       for each eviction: remove from experts HashMap + handle store
       for each load: read_load_source → upload_expert_bundle → insert
       for each route: artifact_fp4_swiglu_ffn_matvec
  → planner.commit_step()
```

### DSV4 layer order

```
hc_pre → attn_norm → attention → hc_post
  → hc_pre → ffn_norm → MoE (routed + shared) → hc_post
```

### What works

- Full 43-layer CUDA greedy generation + readline chat.
- Official DSV4 chat prompt wrapper.
- Packed FP4 + E8M0 scale artifact-preserving expert math.
- Hash routing (early layers) + score/bias routing (later layers).
- HC reference math (`hc_pre`, `hc_post`, `hc_head`, Sinkhorn split).
- Compressed/sliding/indexer attention decode path (correctness-first).
- `deepseek-v4-probe` and `deepseek-v4-generate` CLI commands.

### What's missing

- Official numeric parity (output quality is suspect).
- Device-resident decode arena (host `Vec<f32>` boundaries dominate latency).
- GPU-resident KV/compressor/indexer state.
- Production expert residency + batched selected-expert kernel.
- DSpark speculation.

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
- Context controls: `--ctx-size`, `/reset`, `/stats` in REPL.
- Token/logprob debugging: `--verbose-tokens`, `--logprobs <K>`.
- DSV4 greedy chat uses top-k fast path (no full vocab logits materialization).

---

## Server

Minimal OpenAI-compatible server:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions` (one request at a time)
- SSE streaming (`stream: true`) with token text + final usage stats

Production serving (paged KV, continuous batching, scheduler) is planned after
the graph runtime stabilizes.

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

### llama.cpp parity that matters

| Capability | Ferrule today | Target |
|---|---|---|
| chat CLI | REPL with sampling + stop strings + templates | ✅ structured history + template registry |
| model metadata | `info` | qcache/WeightPack manifest inspection |
| quantization | Q4/Q8/FP4/FP8, mixed precision | GGUF K/IQ execution, calibration |
| benchmarks | `bench-infer` JSON | prompt/decode split, PK matrix |
| quality checks | compare-logits, golden tokens, perplexity | DSV4 official parity |
| loading | safetensors + WeightPack | WeightPack-only, streaming quantization |
| serving | minimal server | paged KV + continuous batching |
| offload | expert streaming (DSV4) | host-staged cache, residency policy |

### Differentiation

Ferrule's differentiation over dense local runtimes:

- **MoE-native**: explicit router/top-k/expert execution, expert hot/cold
  profiling, expert residency and prefetch.
- **Artifact-aware**: preserve official quantization formats (FP4/FP8) instead
  of always re-quantizing.
- **Graph/runtime contracts**: semantic graph IR, typed artifact bindings,
  backend object store.
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
