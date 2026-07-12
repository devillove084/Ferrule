# Ferrule Runtime Graph Architecture

## Status

This document is canonical for Ferrule's **device-independent graph IR**, graph
programs, dialects, validation, and external binding contracts. It is not the
canonical execution-batch, sequence-state, arena, or CUDA graph design; those are
specified in
[`execution-engine-architecture.md`](execution-engine-architecture.md).

The key decision remains: **graph IR is a `ferrule-runtime::graph` module, not a
separate crate**. Ferrule currently keeps a small workspace and uses modules/traits
to draw boundaries.

- `ferrule-runtime::graph` owns device-independent graph IR, `GraphProgram`,
  dialect/shape semantics, and external resource binding contracts.
- E1 deleted the graph-specific execution envelope. The only public execution ABI
  is now `ferrule-common::execution`; graph IR does not define or re-export a second
  `ExecutionBatch` or `ExecutionOutput`.
- `ferrule-runtime` owns sessions, scheduling, storage/residency vocabulary,
  artifact binding, semantic external binding, and lowering from scheduled work to
  the neutral execution ABI. Crate-private `ScheduledBatch` retains runtime
  request/session/KV correlation.
- Reference graph execution receives caller-owned
  `&mut [ReferenceGraphSequenceState]` explicitly; no hidden default sequence state
  or legacy graph execution entrypoint remains.
- Benchmark/report schemas and smoke harnesses live with the CLI/runtime modules that use them; there is no standalone benchmark crate in the current workspace.
- `ferrule-cuda` remains the CUDA backend, kernel, artifact-operator, and CUDA driver graph-capture crate.
- `ferrule-cuda::graph` refers to CUDA driver graph capture/replay, not Ferrule's runtime graph IR.

A future rename from `ferrule-cuda` to something like `ferrule-backend-cuda` may be reasonable, but the CUDA backend should not become the graph IR owner.

A first implementation slice now exists: dense graph translation, generic semantic Transformer graph translation, graph validation, `BackendObjectStore`, and `ReferenceGraphExecutor` with explicit sequence state, plus DSV4 local materialization through semantic artifact groups, stage norm groups, expert registries, and generic config-derived layer policies. The next design pressure is integrating graph preparation with E2 resources/arenas and later expanding coarse semantic `transformer_layer` nodes into fine-grained operator graph nodes without introducing model-family-specific runtime graph files.

## Goals

Ferrule should move from model-specific runners toward a reusable runtime architecture where:

1. Model families describe structure and semantic roles.
2. Graph translation builds a device-independent program.
3. Immutable artifacts, semantic tensor groups, and expert registries are resolved
   through `ExternalBindingPlan` and `BackendObjectStore`.
4. Mutable sequence/KV state is caller-owned and passed explicitly; external
   resource bindings must not become a hidden sequence-state container.
5. Executors consume whole graph programs rather than exposing public per-op traits.
6. CUDA, CPU/reference, and future autograd paths share graph semantics while using
   the sole neutral execution ABI for invocation and explicit state ownership.

This is intended to make Ferrule easier to extend across model families and future training/backprop work without duplicating runners per model family.

## Non-goals

- Do not add a public `FerruleOps` trait with methods such as `linear()`, `rms_norm()`, or `attention()`.
- Do not encode every concrete op as a core `GraphNode` enum variant.
- Do not put CUDA buffers, mmap slices, WeightPack objects, or KV pages directly inside graph nodes.
- Keep the mature DeepSeekV4 eager path as the correctness anchor for future E7
  graph lowering.
- Do not remove the OLMoE CUDA fixture until MoE graph parity is proven.
- Do not rush Qwen3 execution before graph, batch, and binding boundaries are stable.

## High-level architecture

```text
+----------------------------------------------------------------------------------+
|                                  Ferrule CLI                                      |
|  args, chat UX, bench-interactive, CUDA smoke, DSV4 diagnostics                   |
+----------------------------------------+-----------------------------------------+
                                         |
                                         v
+----------------------------------------------------------------------------------+
|                              ferrule-runtime                                     |
|  generation/session/sampler/scheduler/cache                                      |
|  KV policies: contiguous/paged/prefix/radix                                      |
|  storage/residency vocabulary + expert streaming/residency                       |
|  graph-facing contracts:                                                        |
|    - ExternalBindingPlan                                                         |
|    - TransformerSemanticPlan                                                      |
|    - GraphProgram / ReferenceGraphExecutor                                       |
|  execution lowering targets the neutral ABI defined outside graph IR             |
|  mutable reference state is supplied explicitly by the caller                    |
|  Runtime owns algorithms over capabilities; it does not own concrete models.     |
+-------------------------------+-------------------------------+------------------+
                                |                               |
                                v                               v
+-------------------------------------------+   +------------------------------------------+
|              ferrule-model                |   |              ferrule-cuda                |
|  ModelDescriptor / TransformerSpec        |   |  CudaArtifactOperatorContext             |
|  ModelFamily / TensorRole / semantic refs |   |  kernels, device utilities, counters     |
|  artifact binding + HF tensor inventory   |   |  safe smoke benchmarks, CUDA graph        |
|  runner capability traits                 |   |                                          |
|  concrete OLMoE / DeepSeekV4 impls        |   |                                          |
+-------------------------------------------+   +------------------------------------------+
```

## Core graph IR

The core graph intentionally uses opaque operation keys instead of an operator enum.

```rust
pub struct GraphNode {
    id: NodeId,
    op: OpKey,              // opaque: domain + name + version
    inputs: Vec<ValueId>,
    outputs: Vec<ValueId>,
    attrs: AttributeMap,
}
```

The graph core only understands:

- value dependency edges
- graph inputs and outputs
- shape/type/layout metadata
- external handle references
- opaque operation keys and attributes

It does not know how `linear`, `rms_norm`, `attention`, MLA, HC, routing, paged KV, or fused kernels are implemented.

```text
+-----------------------------+
|        ComputeGraph         |
|-----------------------------|
| values: Vec<GraphValue>     |
| nodes:  Vec<GraphNode>      |
| inputs: Vec<ValueId>        |
| outputs: Vec<ValueId>       |
+-------------+---------------+
              |
              v
+-----------------------------+
|          GraphNode          |
|-----------------------------|
| op: OpKey                   |
| inputs: Vec<ValueId>        |
| outputs: Vec<ValueId>       |
| attrs: AttributeMap         |
+-------------+---------------+
              |
              v
+-----------------------------+
|            OpKey            |
|-----------------------------|
| domain:  ferrule.*          |
| name:    dialect-owned      |
| version: u32                |
+-----------------------------+
```

This keeps the graph core generic: the graph is a data structure for dependencies and metadata, while op semantics live in dialect registries and backend lowering code.

## Why opaque nodes instead of `GraphNode` enum variants?

A core enum such as `GraphNode::Linear`, `GraphNode::RmsNorm`, or `GraphNode::Attention` looks convenient at first, but it creates long-term coupling:

- Every new model feature changes the graph crate: MLA, HC, hash routing, sliding window, paged KV, speculation, adapters, fused kernels, etc.
- Every backend must compile against every op variant, even when unsupported.
- Experimental fused ops and backend-only lowerings pollute the public IR.
- Backprop rules get forced into the core enum instead of living next to dialect semantics.

Opaque `OpKey + attrs` avoids this. Concrete op names still exist, but they are dialect data, not public backend trait methods and not core graph enum variants.

## Dialects and registries

Ferrule should group operation semantics into dialects. The exact crate/module placement can evolve, but the graph core should remain independent.

```text
+----------------------+-----------------------------------------------------------+
| Dialect              | Responsibility                                            |
+----------------------+-----------------------------------------------------------+
| ferrule.tensor       | tensor views, layout, elementwise, reductions             |
| ferrule.transformer  | decoder-layer semantics: norm, projection, attention, FFN |
| ferrule.state        | KV cache, paged state, residency, artifact handles        |
| ferrule.moe          | routing, expert selection, expert combine                 |
| ferrule.custom       | experimental, fused, or backend-owned ops                 |
+----------------------+-----------------------------------------------------------+
```

Initial registry traits can stay small:

```rust
pub trait ShapeRegistry {
    fn infer_outputs(
        &self,
        op: &OpKey,
        inputs: &[ValueMeta],
        attrs: &AttributeMap,
    ) -> Result<Vec<ValueMeta>>;
}
```

Later additions can support lowering and training:

```text
ShapeRegistry       op + input metadata + attrs -> output metadata
LoweringRegistry    op -> backend-internal execution recipe
GradientRegistry    op -> backward graph/tape recipe
```

The important rule remains: backend public APIs execute graphs, not individual concrete ops.

## Current execution boundary

The old generic graph backend trait and external-value execution envelope have been
removed. The concrete reference boundary owns its `GraphProgram` and
`BackendObjectStore`, while invocation receives the sole neutral batch and explicit
caller-owned mutable state:

```rust
impl ReferenceGraphExecutor {
    pub fn execute(
        &mut self,
        states: &mut [ReferenceGraphSequenceState],
        batch: &ferrule_common::execution::ExecutionBatch,
    ) -> Result<Vec<TensorData>>;
}
```

```text
+-------------------------+       +----------------------------------+
| GraphProgram            |       | BackendObjectStore               |
| opaque nodes + metadata |       | resolved immutable resources     |
+------------+------------+       +----------------+-----------------+
             |                                     |
             +------------------+------------------+
                                |
+-------------------------------v------------------------------------+
| ReferenceGraphExecutor                                             |
| execute(&mut [ReferenceGraphSequenceState], &ExecutionBatch)       |
+-------------------------------+------------------------------------+
                                |
                                v
                     low-level Vec<TensorData>
```

`ExternalBindingPlan` describes resources; it does not carry mutable sequence state,
runtime request IDs, or scheduler KV handles. `Vec<TensorData>` is a low-level graph
result, not a validated `ModelBatchExecutor::ExecutionOutput`.

The production CUDA target is not a resurrection of a second graph execution API.
Backend preparation may match graph patterns, upload weights, plan arenas, fuse
nodes, and create CUDA graph execs, but invocation still flows through
`PreparedModelPlan + ModelBatchExecutor`, explicit sequence state, and the neutral
`ExecutionBatch` specified by the execution-engine document.

## Graph templates

`GraphTemplate` is a reusable topology recipe. It is not a concrete op trait and does not expose methods such as `linear()` or `attention()`.

```text
+-----------------------------+
|        GraphTemplate        |
|-----------------------------|
| inputs:    TemplateInput[]  |
| externals: TemplateExternal[]|
| constants: TemplateConstant[]|
| nodes:     TemplateNode[]   |
| outputs:   TemplateValueRef[]|
+-------------+---------------+
              |
              | instantiate
              v
+-----------------------------+
|         ComputeGraph         |
+-----------------------------+
```

Templates are useful because decoder models repeat the same topology many times:

```text
Dense decoder block template

  hidden
    |
    v
  attention fragment
    |
    v
  residual
    |
    v
  FFN fragment
    |
    v
  residual output
```

Dense decoder adapters can reuse one dense block template. MoE adapters can reuse a routed-FFN fragment. DeepSeekV4 can later add MLA/HC/hash-routing fragments once dialect support is mature.

## Execution ABI integration — implemented in E1

Graph IR no longer owns an execution envelope. The only public invocation and output
vocabulary is `ferrule-common::execution`, shared by model executors and the
reference graph bridge:

```text
ResidentScheduler
  → SchedulerAction
  → crate-private ScheduledBatch
       ├─ ExecutionBatch: packed tokens/positions/write slots/logits intent
       └─ runtime-only request/session/KV correlation
  → TopKCompatibilityExecutor today
  → native ModelBatchExecutor after E2/E3/E4
```

The public batch uses ragged `ExecutionSequence` spans plus opaque `StateSlot`,
`KvWriteSlot`, and `KvBlockId` values. It contains no `SessionId`, `RequestId`, or
runtime `KvHandle`; output correlation uses `input_row` and the private scheduled
sequence map.

Reference graph execution consumes the same `ExecutionBatch` but receives
`&mut [ReferenceGraphSequenceState]` separately. This prevents graph execution from
silently persisting slot zero or conflating semantic external resources with mutable
sequence/KV state.

The retained serving-shaped path terminates in
`TopKCompatibilityExecutor<R: TopKModelRunner>`. It truthfully supports only one
implicit runner sequence and is not a native multi-session `ModelBatchExecutor`.
E2 prepares reusable resources and arenas, E3 introduces model-native explicit
sequence state, and E4 allows scheduler-formed ragged/multi-session work to reach
one native device pipeline.

## Runtime contract 2: `ExternalBindingPlan`

Graphs should reference runtime-managed objects through semantic external keys, not raw storage objects.

```text
Graph node input
    |
    v
ExternalKey("weights", "layer0.attn.q")
    |
    v
ExternalBindingPlan
    |
    +--> TensorRole::AttentionQuery
    +--> layer: 0
    +--> meta: dtype/shape/layout
    +--> residency: Device / Host / Streamable / Paged / BackendManaged
    |
    v
BackendObjectStore
    |
    +--> CUDA buffer
    +--> CPU tensor
    +--> mmap artifact tensor
    +--> WeightPack object
    +--> resident expert handle
    +--> KV block table / cache handle
```

Advantages:

- Keeps raw Hugging Face tensor names out of graph translators and executors.
- Separates graph structure from storage ownership.
- Lets backends decide upload, residency, eviction, and streamable expert policy.
- Gives weights, KV cache, adapters, speculation state, and resident experts one graph-facing binding model.
- Provides a natural hook for graph-program compilation, memory planning, CUDA graph capture, and future autograd state.

## Model-to-graph translation pipeline

The existing `TransformerSemanticPlan` is the right semantic bridge: it already turns model-family contracts into runtime-level attention, FFN/MoE, KV, and attachment policy without exposing raw tensor names.

```text
+--------------------+
|  ModelDescriptor   |
+---------+----------+
          |
          v
+------------------------+
| ModelSupportContract   |
+---------+--------------+
          |
          v
+------------------------+
| TransformerSemanticPlan |
+---------+--------------+
          |
          v
+------------------------+       +-------------------------+
|   GraphTranslator      +------>|   ExternalBindingPlan   |
+-----------+------------+       +-------------------------+
            |
            v
+------------------------+
| GraphTemplate fragments|
+-----------+------------+
            |
            v
+------------------------+
|      ComputeGraph      |
+-----------+------------+
            |
            v
+------------------------+
|      GraphProgram      |  future cached unit:
| graph + binding +      |  graph, binding plan, shape profile,
| profile + backend art. |  memory plan, compiled/captured artifact
+------------------------+
```

Implementation state and next targets:

1. **Implemented:** dense and generic semantic graph construction plus shape
   validation.
2. **Implemented:** CPU/reference graph execution with explicit per-sequence state.
3. **Implemented baseline:** materialized `BackendObjectStore` and coarse DSV4
   semantic layer binding.
4. **After E2/E3 ownership:** CUDA prepared lowering through the same neutral batch,
   sequence-state, and arena contracts used by eager execution.
5. **Then:** fine-grained dense/MoE/MLA/HC/router dialect coverage and parity without
   model-family-specific runtime graph files.
6. **After native parity:** reusable CUDA graph buckets and fusion; graph capture is
   not a separate execution path.

## Backend lowering model

The CUDA backend should remain an implementation detail behind graph execution.

```text
ComputeGraph
  opaque nodes + attrs + values
        |
        v
+---------------------------+
| CUDA graph backend        |
|---------------------------|
| validate supported ops    |
| resolve ExternalKey       |
| upload/residency policy   |
| allocate scratch buffers  |
| pattern match / fuse      |
| lower to CUDA operators   |
| optional CUDA capture     |
+-------------+-------------+
              |
              v
+---------------------------+
| CudaArtifactOperatorContext |
| CUDA kernels / cuda-oxide |
+---------------------------+
```

This keeps `CudaArtifactOperatorContext` valuable as the existing generic CUDA artifact operator library while preventing low-level kernel methods from becoming the public runtime abstraction.

## Relationship with current code

```text
Current reusable pieces

ferrule-common
  execution.rs                    sole public ExecutionBatch/ExecutionOutput ABI,
                                  capability validation, plan/executor traits

ferrule-model
  spec.rs                         ModelFamily, TransformerSpec, TensorRole
  support/plan.rs                 ModelSupportContract / EnginePlan
  families/qwen3.rs               Qwen3 tensor classification start
  families/deepseek_v4.rs         DeepSeekV4 semantic classification

ferrule-runtime/src/graph
  external_bindings.rs            semantic external resource contracts only
  layer_binding.rs                materialized externals -> layer object bundles
  builder.rs / translate.rs       semantic model descriptors -> graph programs
  template.rs                     reusable graph topology recipes
  shape_registry.rs               graph shape validation support

ferrule-runtime/src/scheduling
  actions.rs                      SchedulerAction and action planning vocabulary
  batch.rs                        crate-private ScheduledBatch correlation/lowering
  session.rs                      GenerateRequest, SequenceState, finish reasons
  resident.rs                     ResidentScheduler over SequenceKvCache

ferrule-runtime/src/cache
  kv.rs                           contiguous + multi-session SequenceKvCache
  paged_kv.rs                     PagedKvCache + PagedSequenceKvCache block tables
  prefix_cache.rs / radix_cache.rs prefix/radix cache direction

ferrule-runtime/src/engine
  worker.rs                       explicit EngineWorker append/decode phases
  topk_compatibility.rs           single-sequence compatibility executor
  driver.rs                       request -> token event -> finish/free-KV loop
  lazy.rs                         artifact-only background load + foreground runner construction

ferrule-runtime
  backend_object_store.rs         materialized HF/semantic externals
  reference_graph_backend.rs      explicit-state ReferenceGraphExecutor
  layer_binding.rs                executable layer state binding
  expert_residency.rs             expert residency backend trait + adapters

ferrule-cuda
  context.rs                      CudaArtifactOperatorContext
  graph.rs                        CUDA driver graph capture/replay
  transformer/*                   current OLMoE CUDA executor pieces
  weightpack.rs                   WeightPack CUDA loading
```

## Ferrule capability map

The immediate goal is not simply to add another model name. The goal is to build the runtime capabilities that make model support scalable inside Ferrule.

| Capability | Current/target Ferrule abstraction |
|---|---|
| Public execution ABI | sole `ferrule-common::execution::ExecutionBatch` and `ExecutionOutput` |
| Current compatibility execution | `ResidentScheduler -> ScheduledBatch -> TopKCompatibilityExecutor` |
| Native continuous batching target | E3/E4 explicit sequence state and native `ModelBatchExecutor` |
| Reusable graph construction | `GraphTemplate -> ComputeGraph -> GraphProgram` |
| External resource separation | `ExternalKey + ExternalBindingPlan + BackendObjectStore` |
| Paged KV direction | runtime reservation plus one physical multi-plane binding lifecycle |
| Artifact/WeightPack loading | external binding plan plus backend object store |
| CUDA fusion/capture | prepared lowering plus persistent arena plus reusable CUDA graph buckets |
| Model-family bring-up | `TransformerSemanticPlan` plus graph translator |
| Future training/backprop | graph IR plus `GradientRegistry`/autograd pass, without a second invocation ABI |

## Backprop and training direction

The same graph architecture should support future backward passes.

```text
Forward ComputeGraph
        |
        v
Autograd pass walks nodes in reverse
        |
        v
GradientRegistry resolves op gradient recipes
        |
        v
Backward graph or tape executor
        |
        v
backend-specific lowering/execution with explicit state
```

This avoids building a separate training abstraction that duplicates inference logic. The forward graph remains the source of truth, while gradient semantics live in dialect registries.

## Graph-IR migration plan

The phases below describe graph-IR construction/materialization. They do not replace
the E0–E8 execution-engine roadmap. Backend graph lowering consumes an already
prepared execution plan and explicit sequence/arena bindings; it must not become a
second execution ownership system.

```text
Phase 0: execution-boundary cleanup — complete
  - Keep graph IR inside ferrule-runtime::graph
  - Use the sole ferrule-common::execution ABI
  - Keep external resource binding separate from mutable sequence state
  - Delete duplicate graph execution entrypoints and implicit default state

Phase 1: graph construction — implemented
  - Build dense decoder templates
  - Generate ComputeGraph and GraphProgram
  - Produce ExternalBindingPlan from semantic roles
  - Validate shapes through the registry

Phase 2: materialized external bridge — implemented baseline
  - Materialize graph externals into BackendObjectStore
  - Aggregate layer-scoped ArtifactGroup / ExpertRegistry / KvState objects
  - Build GraphLayerObjects / executable layer bundles without model-family graph APIs

Phase 3: coarse reference layer execution — implemented baseline
  - Lower coarse transformer_layer nodes into existing runtime components
  - Reuse semantic artifact binding, stage norms, HC, attention, router, MoE, expert streaming, and KV helpers
  - Carry generic attrs/policies for norm eps, HC eps/sinkhorn iters, RoPE/YaRN, compression/indexer metadata, grouped output metadata, SwiGLU limit, route scale, and hash-router layer count
  - Keep explicit ReferenceGraphSequenceState and the neutral ExecutionBatch boundary

Phase 4: fine-grained semantic op split — future graph work
  - Split transformer_layer only after execution ownership is stable
  - Add generic MLA/HC/router/MoE/cache ops and shape inference
  - Keep op names semantic, not model-family-specific

Phase 5: CUDA prepared lowering — after E2/E3 ownership
  - Lower graph recipes internally to CudaArtifactOperatorContext
  - Reuse PersistentArena memory planning and eager *_into stages
  - Add pattern fusion only after native state/batching parity
  - Add reusable CUDA driver graph buckets in E7, not a separate graph path

Phase 6: graph-backed native ModelBatchExecutor — after parity
  - Consume explicit sequence state and the sole neutral batch ABI
  - Keep OLMoE and DeepSeekV4 compatibility paths until native parity exists

Phase 7: serving, speculation, and training/autograd
  - Bind physical paged KV block tables into explicit sequence state
  - Replace the single-sequence TopK compatibility path with native multi-session execution
  - Add DSpark-style speculation through generic Speculation bindings
  - Add GradientRegistry
  - Generate backward graph or tape
  - Reuse the neutral model execution boundary
```

## Open questions

- Should dialect op-key helpers live in `ferrule-runtime`, or should there eventually be a `ferrule-dialects` crate?
- Should shape inference be mandatory during graph construction, or can some backend-specific shapes remain lazy?
- How strict should `ExternalKey` naming be? Prefer `TensorRole + layer + slot` over raw tensor names.
- Should `TensorData` support borrowed or mmap-backed payloads for CPU reference graph execution?
- Should `ferrule-cuda::graph` be renamed to `ferrule-cuda::cuda_graph` to avoid ambiguity with `ferrule-runtime::graph`?
- What is the minimal graph fragment that proves dense decoder coverage without overfitting to one family?
