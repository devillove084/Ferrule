# Ferrule Runtime Graph Architecture

## Status

This document is the canonical design note for Ferrule's graph-facing runtime architecture. It merges the previous compute-graph and runtime-abstraction notes into one plan.

The key decision is: **graph IR is a `ferrule-runtime::graph` module, not a separate crate**. Ferrule currently keeps a small 5-crate workspace and uses modules/traits to draw boundaries.

- `ferrule-runtime::graph` owns the device-independent graph IR, graph programs, graph-facing execution batches, and external binding contracts.
- `ferrule-runtime` also owns sessions, scheduling, storage/residency vocabulary, artifact binding, and semantic external binding.
- Benchmark/report schemas and smoke harnesses live with the CLI/runtime modules that use them; there is no standalone benchmark crate in the current workspace.
- `ferrule-cuda` remains the CUDA backend, kernel, artifact-operator, and CUDA driver graph-capture crate.
- `ferrule-cuda::graph` refers to CUDA driver graph capture/replay, not Ferrule's runtime graph IR.

A future rename from `ferrule-cuda` to something like `ferrule-backend-cuda` may be reasonable, but the CUDA backend should not become the graph IR owner.

A first implementation slice now exists: dense graph translation, generic semantic Transformer graph translation, graph validation, `BackendObjectStore`, `ReferenceGraphBackend`, and DSV4 local materialization through semantic artifact groups, stage norm groups, expert registries, and generic config-derived layer policies. The next design pressure is expanding coarse semantic `transformer_layer` nodes into fine-grained operator graph nodes without introducing model-family-specific runtime graph files.

## Goals

Ferrule should move from model-specific runners toward a reusable runtime architecture where:

1. Model families describe structure and semantic roles.
2. Graph translation builds a device-independent program.
3. Runtime state, weights, KV cache, artifact tensors/groups, expert registries, and resident experts are bound through external handles.
4. Backends execute whole graphs, not public per-op trait methods.
5. CUDA, CPU/reference, and future autograd paths share the same graph-facing contracts.

This is intended to make Ferrule easier to extend across model families and future training/backprop work without duplicating runners per model family.

## Non-goals

- Do not add a public `FerruleOps` trait with methods such as `linear()`, `rms_norm()`, or `attention()`.
- Do not encode every concrete op as a core `GraphNode` enum variant.
- Do not put CUDA buffers, mmap slices, WeightPack objects, or KV pages directly inside graph nodes.
- Do not remove the mature DeepSeekV4 path before graph parity exists.
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
|    - ExecutionBatch / ExecutionOutput                                            |
|    - ExternalBindingPlan                                                         |
|    - TransformerRuntimePlan                                                      |
|    - GraphProgram / GraphRunner                                                  |
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

## Backend boundary

Backends receive a whole graph plus external bindings and runtime inputs.

```rust
pub trait GraphBackend {
    fn execute(
        &mut self,
        graph: &ComputeGraph,
        external: &ExternalHandles,
        inputs: &[TensorData],
    ) -> Result<Vec<TensorData>>;
}
```

```text
+------------------+       +----------------------+       +----------------------+
|  ComputeGraph    |       |   ExternalHandles    |       |      TensorData      |
|  opaque nodes    |       |  weights / KV / etc  |       |  runtime inputs      |
+---------+--------+       +----------+-----------+       +----------+-----------+
          |                           |                              |
          +---------------------------+------------------------------+
                                      |
                                      v
                         +--------------------------+
                         |      GraphBackend        |
                         | execute(graph, ...)      |
                         +------------+-------------+
                                      |
                +---------------------+----------------------+
                |                     |                      |
                v                     v                      v
       +----------------+     +----------------+      +----------------+
       | CPU reference  |     | CUDA backend   |      | Autograd       |
       | interpreter    |     | lower + fuse   |      | tape/backward  |
       +----------------+     +----------------+      +----------------+
```

The CUDA backend can internally match graph patterns, fuse nodes, plan memory, upload weights, and capture CUDA driver graphs. None of those decisions need to leak into the `GraphBackend` trait.

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

## Runtime contract 1: `ExecutionBatch` / `ExecutionOutput`

`ExecutionBatch` is Ferrule's graph-facing execution envelope. It groups token, position, session, KV, and logits-selection metadata into one runtime input contract. `ExecutionOutput` is the paired result envelope: each output row carries the session, position, KV handle, and optional full/top-k logits.

```text
+-----------------------------+
|       ExecutionBatch        |
|-----------------------------|
| segment: Prefill/Decode/Mix |
| rows: Vec<ExecutionRow>     |
+-------------+---------------+
              |
              v
+-----------------------------+
|        ExecutionRow         |
|-----------------------------|
| token_id: u32               |
| position: usize             |
| session_id: SessionId       |
| kv_handle: Option<KvHandle> |
| require_logits: bool        |
+-----------------------------+
```

Advantages:

- Makes token ids, positions, sessions, KV handles, and logits selection explicit.
- Allows schedulers to form multi-row decode, chunked prefill, or mixed prefill/decode batches.
- Avoids producing logits for rows that do not need them.
- Gives graph runners a stable input contract without replacing legacy `ModelRunner` immediately.
- Opens the path to continuous batching and paged KV scheduling.

Legacy runners can keep using `prefill(tokens)` and `decode_token(token)`. Graph-backed runners can opt into `ExecutionBatch` / `ExecutionOutput` when ready. `SchedulerAction` is the current bridge from resident sequence state to concrete prefill/decode batches, `ResidentActionExecutor` is the capability-based adapter that executes those actions against a single resident `TopKModelRunner`, and `ResidentTopKDriver` closes the loop with token events, EOS/stop/max-token finish policy, and KV release without depending on any concrete model family.

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

The existing `TransformerRuntimePlan` is the right semantic bridge: it already turns model-family contracts into runtime-level attention, FFN/MoE, KV, and attachment policy without exposing raw tensor names.

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
| TransformerRuntimePlan |
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

Recommended first targets:

1. Dense decoder graph generation, without rushing execution.
2. CPU/reference graph executor for correctness.
3. CUDA graph backend adapter using `CudaArtifactOperatorContext` internally.
4. Qwen3 dense execution once graph/batch/binding contracts are validated.
5. MoE graph parity using OLMoE/Mixtral/Qwen-MoE style fragments.
6. DeepSeekV4 migration only after MLA/HC/hash-routing dialect coverage is strong.

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

ferrule-model
  spec.rs                         ModelFamily, TransformerSpec, TensorRole
  support/plan.rs                 ModelSupportContract / EnginePlan
  families/qwen3.rs               Qwen3 tensor classification start
  families/deepseek_v4.rs         DeepSeekV4 semantic classification

ferrule-runtime/src/graph
  runtime.rs                      ExecutionBatch, ExecutionOutput, ExternalBindingPlan
  layer_binding.rs                materialized externals -> layer object bundles
  builder.rs / translate.rs       semantic model descriptors -> graph programs
  template.rs                     reusable graph topology recipes
  shape_registry.rs               graph shape validation support

ferrule-runtime/src/scheduling
  session.rs                      GenerateRequest, SequenceState, finish reasons
  scheduler.rs                    SchedulerAction, PrefillChunkAction, DecodeAction
  resident.rs                     ResidentScheduler over SequenceKvCache

ferrule-runtime/src/cache
  kv.rs                           contiguous + multi-session SequenceKvCache
  paged_kv.rs                     PagedKvCache + PagedSequenceKvCache block tables
  prefix_cache.rs / radix_cache.rs prefix/radix cache direction

ferrule-runtime/src/engine
  worker.rs                       EngineWorker phased append/decode API
  executor.rs                     SchedulerAction -> ExecutionOutput adapter over TopKModelRunner
  driver.rs                       request -> token event -> finish/free-KV loop
  lazy.rs                         artifact-only background load + foreground runner construction

ferrule-runtime
  backend_object_store.rs         materialized HF/semantic externals
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

```text
+-------------------------------+----------------------------------------------+
| Capability                    | Ferrule abstraction                           |
+-------------------------------+----------------------------------------------+
| batched graph execution       | ExecutionBatch                                |
| reusable graph construction   | GraphTemplate -> ComputeGraph                 |
| backend buffer separation     | ExternalKey + ExternalBindingPlan             |
| paged KV direction            | KvHandle + PagedSequenceKvCache + KvState     |
| continuous batching           | ResidentScheduler + ResidentTopKDriver + ExecutionBatch |
| artifact/WeightPack loading     | ExternalBindingPlan + backend object store    |
| CUDA fusion/capture           | backend lowering + memory plan + CUDA capture |
| model-family bring-up         | TransformerRuntimePlan + graph translator     |
| future training/backprop      | graph IR + GradientRegistry / autograd pass   |
+-------------------------------+----------------------------------------------+
```

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
GraphBackend::execute(...)
```

This avoids building a separate training abstraction that duplicates inference logic. The forward graph remains the source of truth, while gradient semantics live in dialect registries.

## Migration plan

```text
Phase 0: cleanup and skeletons
  - Remove legacy OLMoE CPU forward/KV path
  - Keep OLMoE CUDA fixture
  - Keep graph IR inside ferrule-runtime::graph
  - Add graph_runtime.rs with ExecutionBatch and ExternalBindingPlan

Phase 1: graph construction
  - Build dense decoder templates
  - Generate ComputeGraph without executing it
  - Produce ExternalBindingPlan from semantic roles
  - Add shape validation registry

Phase 2: materialized external bridge
  - Materialize graph externals into BackendObjectStore
  - Aggregate layer-scoped ArtifactGroup / ExpertRegistry / KvState objects
  - Build GraphLayerObjects / executable layer bundles without model-family graph APIs

Phase 3: coarse semantic layer execution
  - Lower coarse transformer_layer nodes into existing runtime components
  - Reuse semantic artifact binding, stage norms, HC, attention, router, MoE, expert streaming, and KV helpers
  - Carry generic attrs/policies for norm eps, HC eps/sinkhorn iters, RoPE/YaRN, compression/indexer metadata, grouped output metadata, SwiGLU limit, route scale, and hash-router layer count
  - Prove tiny/synthetic layer execution, then local DSV4 layer execution when present

Phase 4: fine-grained semantic op split
  - Split transformer_layer after the coarse bridge executes
  - Add generic MLA/HC/router/MoE/cache ops and shape inference
  - Keep op names semantic, not model-family-specific

Phase 5: CUDA graph backend adapter
  - Lower graph nodes internally to CudaArtifactOperatorContext
  - Add scratch allocation and memory planning
  - Add pattern fusion
  - Add optional CUDA driver graph capture

Phase 6: graph-backed RuntimeRunner
  - Add RuntimeRunner::Graph(...)
  - Keep OLMoE and DeepSeekV4 legacy paths until graph-backed parity exists

Phase 7: serving, speculation, and training/autograd
  - Bind paged KV block tables into graph/backend KvState execution
  - Add multi-session backend execution beyond the single-runner TopK adapter
  - Add DSpark-style speculation through generic Speculation bindings
  - Add GradientRegistry
  - Generate backward graph or tape
  - Reuse backend execute boundary
```

## Open questions

- Should dialect op-key helpers live in `ferrule-runtime`, or should there eventually be a `ferrule-dialects` crate?
- Should shape inference be mandatory during graph construction, or can some backend-specific shapes remain lazy?
- How strict should `ExternalKey` naming be? Prefer `TensorRole + layer + slot` over raw tensor names.
- Should `TensorData` support borrowed or mmap-backed payloads for CPU reference graph execution?
- Should `ferrule-cuda::graph` be renamed to `ferrule-cuda::cuda_graph` to avoid ambiguity with `ferrule-runtime::graph`?
- What is the minimal graph fragment that proves dense decoder coverage without overfitting to one family?
