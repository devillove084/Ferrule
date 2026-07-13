# Ferrule Architecture

Ferrule is a Rust-native, state-aware LLM runtime for edge inference. It targets
sparse MoE inference under memory pressure, with **OLMoE** as the correctness
golden model and **DeepSeek V4 Flash / DSpark** as the near-term pressure test.

The thesis: inference, rollout, quantization, cache management, and future
training should share one native systems foundation. Router decisions, selected
experts, quantized weights, KV cache, expert residency, and rollout state should
be explicit runtime objects that can be scheduled against real hardware.

Canonical execution documents:

- [`execution-engine-architecture.md`](execution-engine-architecture.md) defines
  `ExecutionBatch`, prepared plans, sequence state, persistent arenas, physical KV,
  residency ownership, and eager/graph unification.
- [`ROADMAP.md`](ROADMAP.md) defines the E0–E8 implementation order and gates.
- [`status/2026-07-11-gb10-dsv4.md`](status/2026-07-11-gb10-dsv4.md) records the
  current verified DSV4 correctness/performance baseline.

---

## Table of Contents

1. [Crate layout](#crate-layout)
2. [Current runtime](#current-runtime)
3. [Execution engine target](#execution-engine-target)
4. [Runtime graph](#runtime-graph)
5. [Storage and residency](#storage-and-residency)
6. [Model-family boundary](#model-family-boundary)
7. [OLMoE forward pass](#olmoe-forward-pass)
8. [GPU runtime](#gpu-runtime)
9. [DeepSeek V4 execution](#deepseek-v4-execution)
10. [Quantization and WeightPack](#quantization-and-weightpack)
11. [KV cache](#kv-cache)
12. [Sampling and chat](#sampling-and-chat)
13. [Serving](#serving)
14. [Composable engine architecture](#composable-engine-architecture)
15. [Alignment targets](#alignment-targets)
16. [Future: Elastic State Fabric](#future-elastic-state-fabric)
17. [Future: training and RL](#future-training-and-rl)

---

![Ferrule architecture](assets/ferrule-current-architecture.svg)

## Crate layout

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ferrule-cli                                  │
│  args · chat · serve composition root · benchmarks · diagnostics     │
└───────────────┬──────────────────────────────────────┬──────────────┘
                │                                      │
                ▼                                      ▼
┌──────────────────────────────┐      ┌───────────────────────────────┐
│        ferrule-server        │      │       direct diagnostics       │
│ Axum/Hyper/Tokio · OpenAI/SSE│      │ terminal/JSON over runtime     │
│ bounded channels · worker    │      └───────────────┬───────────────┘
└───────────────┬──────────────┘                      │
                └──────────────────┬───────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ferrule-runtime                                 │
│  generation/session/sampling/scheduler/cache                        │
│  graph IR + private ScheduledBatch correlation/lowering             │
│  storage vocabulary + expert residency coordinator/policy            │
│  Owns algorithms over capability traits; owns no concrete model.     │
└───────────────┬───────────────────────────────────┬─────────────────┘
                │                                   │
                ▼                                   ▼
┌──────────────────────────────┐     ┌───────────────────────────────┐
│         ferrule-model        │     │          ferrule-cuda          │
│  model semantics             │     │  CUDA primitives/kernels       │
│  artifact + semantic binding │     │  device utilities/counters     │
│  runner capability traits    │     │  safe smoke benchmark API      │
│  concrete OLMoE / DSV4 impls │     │  unsafe launch hidden here     │
│  tokenizer + HF inventory    │     └───────────────────────────────┘
└───────────────┬──────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ferrule-common                                  │
│  shared error/result, sole execution + expert-residency neutral ABIs  │
└─────────────────────────────────────────────────────────────────────┘
```

| Crate | Role |
|---|---|
| `ferrule-common` | shared errors, `QuantType`, the sole dependency-neutral execution ABI, and the model-neutral expert slot/generation/lease control ABI |
| `ferrule-model` | model semantics, artifact binding, prepared model/lowering policy, CPU reference, and concrete model implementations such as `models::deepseek_v4` |
| `ferrule-runtime` | generation/session/sampling/scheduling, logical sequence/KV lifecycle, runtime graph IR, storage/residency and executable-cache policy; depends on capabilities, not concrete models |
| `ferrule-cuda` | CUDA allocations/pools, streams/events, primitives/kernels, physical device resources, counters, and CUDA graph execs |
| `ferrule-server` | model-neutral Axum/Hyper/Tokio OpenAI API, SSE serialization, bounded admission/token channels, dedicated model-worker ownership, and disconnect cancellation |
| `ferrule-cli` | argument parsing, command dispatch, model-family composition roots, terminal/JSON output, and diagnostics |

---

## Current runtime

```text
safetensors + tokenizer files
        ↓
ferrule-model family descriptor + tensor inventory
        ↓
Artifact-preserving model payloads
        ↓
ferrule-model concrete runner implements capability traits
        ↓
ferrule-runtime scheduler/KV lifecycle + per-layer expert residency controller
        ↓
model runner sequence state + CUDA physical pages/expert handles/stable tables
        ↓
chat / one-shot generation / diagnostics
```

Two execution paths coexist:

1. **OLMoE all-resident path**: `GpuOlmoeModel::build_from_cpu` quantizes and
   uploads all experts at startup as concatenated device buffers. It is the
   correctness fixture for router semantics and legacy CUDA kernels.

2. **DSV4 interactive streaming path**: `ferrule-model::models::deepseek_v4`
   owns model semantics, immutable expert source catalogs, the compatibility default
   sequence, and concrete lowering. `KvPageManager` owns logical page allocation,
   refcounts, block tables, COW, and preemption; the CUDA backend owns physical page
   planes plus expert staging/upload tickets, handles, and stable tables. The
   runtime-injected per-layer `ExpertResidencyController` owns logical expert slots,
   generations, leases, LRU/admission, and cross-request stats. CPU/reference DSV4
   alone retains `ExpertStreamingPlanner` and `CpuExpertHandleStore`.

Current DSV4 execution is a **single-process resident runner**, not yet a production
serving engine. Two UX paths exist:

- `ferrule chat ... -q cuda --chat-template deepseek-v4 --temp 0` starts the REPL
  immediately and loads CPU-side DSV4 artifacts on a background thread.
- `bench-interactive` runs real full 43-layer DSV4 through `ResidentTopKDriver`,
  proving request admission, token-budgeted ragged/mixed scheduling, packed execution,
  token events, finish reasons, and authoritative physical KV release.
- CUDA context/operator cache is initialized on the main thread when the model is
  first used.
- Interactive prompt append uses `prefill_tokens_topk_interactive`, now routed
  through the DSV4 batched/segment prefill core. Runtime non-final chunks call
  `TopKModelRunner::prefill_tokens`, so they update session/KV/MoE state without
  materializing hidden/logits; the final chunk materializes top-k for generation.
- `KvPageManager` is the sole logical block-table/refcount owner; DSV4 binds its
  model-neutral schema to preallocated CUDA multi-plane page pools. Reserve, execute,
  commit/rollback, COW, preempt/restore, and exact-prefix sharing use one lifecycle.
- Packed decode, ragged prefill, and mixed prefill/decode share one rows pipeline.
  Score/hash routing, compact grouping, stable-slot resolution, and residency
  publication are device-side; runtime owns residency policy. Stable graph buckets
  remain E7 work.

---

## Execution engine target

The next architecture is not another runner façade. It separates four lifetimes:

```text
PreparedModelPlan          immutable model-global recipes and typed handles
SequenceExecutionState     one logical sequence's committed model/KV state
PersistentArena            reusable phase/shape-bucket scratch and metadata
ExecutionBatch             one packed/ragged invocation over explicit states
```

Runtime lowers scheduled requests into `ExecutionBatch`; a prepared executor
borrows explicit sequence states and an arena lease; CUDA owns physical buffers,
streams/events, pages, expert slots, and graph execs. Eager and graph modes call the
same allocation-free device stages.

E1–E3 implemented the neutral `ExecutionBatch`, prepared resources and reusable
arenas, and model-native per-sequence semantic/physical state with explicit
lifecycle. Native ragged/multi-session execution follows in E4. The DSV4 state split,
arena contents, unified device pipeline, paged multi-plane KV, device
routing/residency, graph buckets, and module migration are specified in
[`execution-engine-architecture.md`](execution-engine-architecture.md).
Implementation order and hard gates are in [`ROADMAP.md`](ROADMAP.md).

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

Current shape: coarse — one `transformer_layer` node per layer. The reference graph
bridge requires caller-owned state and the sole neutral execution ABI:

```
GraphProgram + BackendObjectStore
  + &mut [ReferenceGraphSequenceState]
  + &ferrule_common::execution::ExecutionBatch
  → ReferenceGraphExecutor::execute(...)
  → GraphLayerObjects (aggregated by ArtifactGroupKind + layer)
  → typed artifact binders and existing attention/HC/MoE/KV components
  → Vec<TensorData>
```

The low-level tensor result is not `ModelBatchExecutor::ExecutionOutput`, and the
reference executor does not hide mutable state in an implicit default slot.

Future graph-IR work may split `transformer_layer` into fine-grained semantic ops
(`layer_hc_pre`, `rms_norm`, `latent_attention`, `router_select`, `routed_moe`,
`shared_ffn`, `residual_merge`). This is independent of the immediate execution
ownership work: graph lowering must consume the neutral prepared-executor ABI rather
than create a second sequence/KV lifecycle. See
[`runtime-graph-architecture.md`](runtime-graph-architecture.md) and
[`execution-engine-architecture.md`](execution-engine-architecture.md).

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

- `ferrule-runtime::storage` provides backend-neutral vocabulary:
  `StorageObjectId`, `ObjectLocator`, `Placement`, `ObjectReplica`,
  `ReplicaHandleId`, `StorageCatalog`, `TransferEngine`, and policy scoring
  types.
- `ferrule-common::ExpertResidencyControl` defines model-qualified expert identity,
  stable slot/generation bindings, selected leases, and prepare/publish/cancel.
  `ferrule-runtime::ExpertResidencyController` is the active CUDA residency policy:
  one coordinator per layer owns capacity, deterministic LRU, reservations, leases,
  and stats. `ExpertStreamingPlanner` remains only in CPU/reference execution.
- `HostStagedExpertCache` caches decoded expert bundles in host RAM and uses mmap
  shard reads underneath.
- CUDA DSV4 selected and prefetch misses use pinned asynchronous uploads and
  compute-stream event dependencies. Stable-table publication/eviction uses device
  kernels with no normal-path table H2D copy or stream-wide synchronization; runtime
  capacity prevents unbounded physical residency.
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

Current (transitional) DSV4 CUDA state is owned by `DeepSeekV4CudaOperatorCache`.
This is intentionally not the target ownership boundary: it still mixes prepared
resources, one sequence's physical state, expert handles, scratch, and diagnostics.
The split into `PreparedModelPlan`, `SequenceExecutionState`, and `PersistentArena`
is specified in [`execution-engine-architecture.md`](execution-engine-architecture.md).

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
| `moe_gemv_*_batched` | legacy FP4 scalar/reduction diagnostic primitives; not a complete end-to-end fallback |
| `rms_norm_*`, `rope_tail_from_device` | device norm/rotary pieces |
| `sparse_attention_sink_from_device` | sparse attention over device query/value buffers |
| `artifact_linear_topk_from_device` | greedy output-head top-k without full logits download |

The FP4 Tensor Core path is correct and default-on. `FERRULE_CUDA_MOE_TC=0` is a
legacy decode diagnostic that currently rejects batched prefill, not a complete
scalar fallback. Decode still often has one real token column; sustained GEMM
utilization requires native multi-sequence/prefill columns and device-side
per-column routing.

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

Interactive chat avoids final logits for non-final prompt work and routes fresh
prompt prefill through the batched DSV4 core. The final prompt row materializes
requested top-k only when generation needs it.

This is still not whole-model device-resident chunked serving prefill: HC rows cross
host memory between layers, and appended segments can fall back to token-wise decode
execution. Native ragged prefill is an E4 target.

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
- Managed-memory FP4 expert residency with mmap, host-staged, pinned, and in-flight
  upload mechanisms.
- Device-resident single-token layer execution and per-layer batched prefill.
- FP4 `mxf4` Tensor Core MoE path, grouped prefill columns, and deterministic
  route-ranked device reduction.
- GPU compressor softmax/RMS/RoPE/QAT numerics matching token-loop execution.
- Typed growable RoPE resources and failure-atomic combined-KV growth.
- Full 43-layer batched/token-loop bit-exact parity and stable prefill→decode
  continuation.
- Routing/residency counters, JSON benchmark attribution, and layer/attention
  parity checkpoints.

`FERRULE_CUDA_MOE_TC=0` is a legacy decode diagnostic and currently rejects
batched prefill; it is not a complete production scalar fallback.

### Measured current behavior

Current GB10 cold correctness baseline for the five-token chat prompt:

- prefill: approximately **17.47–17.75s**;
- decode: approximately **0.80–0.824 tok/s**;
- first token: approximately **18.76–19.01s**;
- generated continuation: `[30594, 1175]`.

Three independent 43-layer parity runs report every layer `max_abs_diff=0.0`.
Detailed commands and results are in
[`status/2026-07-11-gb10-dsv4.md`](status/2026-07-11-gb10-dsv4.md).

### What's missing

- Whole-model device-resident prefill HC/hidden ping-pong; current layer boundaries
  still cross host memory.
- Explicit prepared plan, persistent arena, and per-sequence execution state.
- Native multi-session/ragged/mixed execution instead of a single implicit runner
  session.
- Physical CUDA paged multi-plane KV, prefix/radix reuse, and preemption.
- Fully device-resident compressor metadata, router/grouping, and stable expert
  indirection under a runtime residency coordinator.
- E7 long-lived piecewise CUDA graph buckets; DSV4 currently has no CUDA graph
  execution mode and always uses device-resident eager decode.
- Rollback/KV rewind for cancellation and speculative branches.
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

- `KvPageManager`: serving authority for logical allocation, free lists, refcounts,
  block tables, generations, reservations, rollback, COW, and preempt/restore.
- `CudaKvPagePool`: bounded physical multi-plane CUDA storage with compact block-table
  lowering; growth allocates pages and never copies full history.
- DSV4 paged window/main/indexer scatters, indexer top-k, and single/dual-plane sparse
  attention consume active page bindings directly.
- `ContiguousKvCache`, `MultiSessionKvCache`, `PagedKvCache`, and
  `PagedSequenceKvCache` remain CPU/reference test infrastructure, not production
  DSV4 CUDA ownership.
- Exact-prefix serving forks share pages and use partial-tail COW. Radix lookup
  integration remains future work.
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

`ResidentScheduler` owns request admission and logical `SequenceState`; runtime
lowers token-budgeted `SchedulerAction`s into a crate-private `ScheduledBatch` that
preserves request/session/KV correlation. `NativeMultiSessionExecutor` consumes the
neutral `ExecutionBatch`, and `ResidentTopKDriver` coordinates explicit model states
with `KvPageManager` and backend physical-page transactions. Chat retains resident
sessions across turns; token-loop execution exists only inside the page-managed
differential diagnostic harness.

E1–E6 now provide the execution ABI, prepared resources/arenas, explicit sequence
state, native packed multi-session execution, authoritative physical paged KV,
device routing/grouping, and runtime-owned expert residency. E7–E8 still own stable
CUDA graphs, fusion, device sampling, and competitive serving validation. See
[`execution-engine-architecture.md`](execution-engine-architecture.md).

---

## Serving

Serving is active through the model-neutral `ferrule-server` crate and the DSV4
`ferrule serve` composition root. The implemented OpenAI-compatible surface is:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- streaming and non-streaming responses, final usage, and SSE `[DONE]`

Axum/Hyper/Tokio handlers submit through a bounded channel to one dedicated model
thread. That thread constructs and exclusively owns the tokenizer, CUDA runner,
`ResidentTopKDriver`, scheduler, and paged KV state. Per-request bounded token
channels prevent a slow client from blocking every sequence. Closing an SSE response
sets an atomic cancellation flag; the owner thread cancels at the next model-step
boundary and releases runtime/model/KV state without returning a callback error or
poisoning unrelated requests.

The API currently accepts only truthful greedy generation (`temperature = 0`,
`n = 1`, top-k 1) and rejects unsupported sampling/logprob/tool fields. E7 graph
buckets, E8 device sampling/fusion, automatic radix-prefix lookup, metrics endpoints,
and official vLLM/SGLang benchmark publication remain. See
[`serving.md`](serving.md) for the contract and commands.

---

## Composable engine architecture

The engine first composes model-family policies into an `EnginePlan`, which is a
support/compatibility report rather than an executable instance:

```text
ModelDescriptor → ModelLayout → ModelSupportContract → EnginePlan {
  ModelFamilyPolicy, AttentionPolicy, RouterPolicy, ExpertPolicy,
  QuantPolicy, KvPolicy, SchedulerPolicy, ParallelismPlan,
  ResidencyPolicy, SpeculationPolicy
}
```

A later `PreparedModelPlan` binds that report to validated immutable artifacts,
typed backend handles, a KV schema, and supported execution capabilities. Dynamic
sequence bindings and batch shapes do not belong to either plan; they belong to
`SequenceExecutionState`, `ExecutionBatch`, and `PersistentArena` leases.

Configuration switches (`engine`, `attention`, `kv`, `parallel`, `residency`,
`speculation`) map to typed policy objects. Model-specific assumptions must not
leak into scheduler, sampler, CLI, or the neutral execution ABI.

| Policy | Current evidence | Execution target |
|---|---|---|
| `AttentionPolicy` | OLMoE GQA; DSV4 MLA/sparse attention and compressed/indexer semantics | prepared attention recipes and physical multi-plane paged KV |
| `RouterPolicy` | dense top-k plus DSV4 hash and score/bias routing | device top-k, token/expert grouping, EP only after local batching is real |
| `ExpertPolicy` | routed/shared DSV4 experts with grouped prefill kernel path | packed multi-sequence grouped execution and stable expert indirection |
| `QuantPolicy` | Q4/Q8 WeightPack; DSV4 FP4/FP8 artifact path | calibration/quality gates; new formats only after hot-path ownership stabilizes |
| `KvPolicy` | legacy contiguous KV plus scheduler-facing paged metadata | physical CUDA paged multi-plane KV, prefix/COW/preemption |
| `SchedulerPolicy` | resident action scheduling over request/session metadata | native ragged prefill, multi-row decode, fairness and preemption |
| `ResidencyPolicy` | runtime-owned per-layer coordinator, leases, generations, stats; DSV4 owns staged/pinned physical mechanisms | integrate stable bindings with E7 graph buckets and later distributed placement |
| `SpeculationPolicy` | DSpark/MTP attachment metadata | isolated proposal/verify/rollback after sequence and KV rollback are safe |

---

## Alignment targets

### Mainstream-engine parity that matters

| Capability | Ferrule today | Target |
|---|---|---|
| Correctness | real DSV4 43L batched/token-loop is bit-exact; continuation `[30594,1175]` | preserve oracle through every ownership and kernel migration; add long-context, graph, failure, and multi-sequence gates |
| Execution ABI | sole packed/ragged `ExecutionBatch`, private `ScheduledBatch`, and native `MultiSessionRunner`/`NativeMultiSessionExecutor` | preserve the neutral ABI through E7–E8 |
| Prompt prefill | native paged ragged/mixed CUDA rows pipeline; token loop is diagnostic-only; device router/residency is complete | stable graph buckets and further fusion |
| Decode | native packed multi-row paged decode with exact batch-2/4 serial parity and device routing/residency | stable graph buckets and device sampling |
| KV | authoritative physical paged multi-plane lifecycle with reservation/COW/prefix/preemption | radix lookup policy and later low-bit KV formats |
| MoE residency | runtime cross-request coordinator, exact stable bindings/leases/stats, and event-guarded CUDA physical resources | E7 graph binding plus later distributed placement |
| CUDA graph | no current DSV4 graph mode; device-resident eager decode only | E7 long-lived piecewise buckets over the eager device pipeline |
| Serving | active OpenAI chat/completion HTTP/SSE, bounded async admission/fanout, dedicated model owner, per-request EOS, and disconnect cancellation | official benchmark validation, metrics, device sampling, auth/deployment hardening |
| Quantization | Q4/Q8/FP4/FP8, mixed precision, artifact-preserving DSV4 | calibrated quality gates and additional formats only after execution ownership stabilizes |

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
