# Ferrule Runtime Graph Architecture

## Status

This document is the canonical design note for Ferrule's graph-facing runtime architecture. It merges the previous compute-graph and runtime-abstraction notes into one plan.

The key decision is: **do not rename `ferrule-cuda` to `ferrule-graph`**.

- `ferrule-graph` is the device-independent graph IR crate.
- `ferrule-runtime` owns sessions, scheduling, source binding, semantic external binding, and graph-facing execution batches.
- `ferrule-cuda` remains the CUDA backend, kernel, source-operator, WeightPack, and CUDA driver graph-capture crate.
- `ferrule-cuda::graph` refers to CUDA driver graph capture/replay, not Ferrule's compute graph IR.

A future rename from `ferrule-cuda` to something like `ferrule-backend-cuda` may be reasonable, but the CUDA backend should not become the graph IR crate.

## Goals

Ferrule should move from model-specific runners toward a reusable runtime architecture where:

1. Model families describe structure and semantic roles.
2. Graph translation builds a device-independent program.
3. Runtime state, weights, KV cache, source tensors, and resident experts are bound through external handles.
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
|  commands, chat/server entrypoints, benchmarks, probes, local smoke tests         |
+----------------------------------------+-----------------------------------------+
                                         |
                                         v
+----------------------------------------------------------------------------------+
|                              ferrule-runtime                                     |
|                                                                                  |
|  generation/chat/sampler/session/scheduler                                       |
|  KV cache policies: contiguous, paged, prefix, radix                             |
|  source binding: SourceTensorPayload, SourceLinearPayload, LayerSourceBinding     |
|  reference anchors: attention, FFN, routing, MoE, HC                              |
|                                                                                  |
|  Graph-facing contracts:                                                         |
|    - ExecutionBatch                                                              |
|    - ExternalBindingPlan                                                         |
|    - TransformerRuntimePlan                                                      |
|    - future GraphProgram / GraphRunner                                           |
+--------------------------+-------------------------------+-----------------------+
                           |                               |
                           v                               v
+----------------------------------------+     +-----------------------------------+
|              ferrule-model             |     |           ferrule-graph           |
|                                        |     |                                   |
|  ModelDescriptor                       |     |  ComputeGraph                    |
|  TransformerSpec                       |     |  GraphTemplate                   |
|  ModelSupportContract                  |     |  GraphNode { OpKey, attrs }      |
|  ModelFamily                           |     |  ValueMeta / TensorShape         |
|  TensorRole                            |     |  ExternalKey / ExternalHandles   |
|  HF/GGUF tensor classification         |     |  GraphBackend::execute(...)      |
+----------------------------------------+     +----------------+------------------+
                                                                |
                                                                v
+----------------------------------------------------------------------------------+
|                                Backend crates                                     |
|                                                                                  |
|  ferrule-cuda                                                                    |
|    - CudaSourceOperatorContext                                                   |
|    - CUDA kernels and cuda-oxide integration                                      |
|    - WeightPack CUDA loading                                                     |
|    - CUDA driver graph capture/replay                                             |
|    - future CudaGraphBackend                                                     |
|                                                                                  |
|  future CPU/reference graph backend                                               |
|    - correctness-first interpreter over graph nodes                               |
|                                                                                  |
|  future Autograd backend                                                          |
|    - backward graph/tape built from dialect gradient registries                   |
+----------------------------------------------------------------------------------+
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
| ferrule.state        | KV cache, paged state, residency, source handles          |
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

## Runtime contract 1: `ExecutionBatch`

`ExecutionBatch` is Ferrule's graph-facing execution envelope. It groups token, position, session, KV, and logits-selection metadata into one runtime input contract.

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

Legacy runners can keep using `prefill(tokens)` and `decode_token(token)`. Graph-backed runners can opt into `ExecutionBatch` when ready.

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
    +--> mmap source tensor
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
3. CUDA graph backend adapter using `CudaSourceOperatorContext` internally.
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
| CudaSourceOperatorContext |
| CUDA kernels / cuda-oxide |
+---------------------------+
```

This keeps `CudaSourceOperatorContext` valuable as the existing generic CUDA source operator library while preventing low-level kernel methods from becoming the public runtime abstraction.

## Relationship with current code

```text
Current reusable pieces

ferrule-model
  spec.rs                         ModelFamily, TransformerSpec, TensorRole
  support/plan.rs                 ModelSupportContract / EnginePlan
  families/qwen3.rs               Qwen3 tensor classification start
  families/deepseek_v4.rs         DeepSeekV4 semantic classification

ferrule-runtime
  graph_runtime.rs                ExecutionBatch, ExternalBindingPlan
  transformer_plan.rs             semantic model plan -> runtime plan
  source_tensor.rs                bounded source tensor payload/readers
  source_linear.rs                source-format linear payload + CPU reference
  source_binding.rs               semantic source binding
  attention_backend.rs            sparse/reference attention anchors
  ffn.rs                          SwiGLU payload/reference pieces
  expert_routing.rs               routing policies
  routed_moe.rs                   routed MoE reference execution
  hyper_connection.rs             HC reference anchors
  kv.rs / paged_kv.rs             KV cache policy pieces
  prefix_cache.rs / radix_cache.rs prefix/radix cache direction

ferrule-graph
  ComputeGraph                    opaque IR
  GraphTemplate                   reusable topology recipe
  GraphBackend                    whole-graph backend boundary
  ExternalKey / ExternalHandles   graph-facing external references

ferrule-cuda
  context.rs                      CudaSourceOperatorContext
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
| paged KV direction            | KvHandle + ExternalBindingKind::KvState       |
| continuous batching           | ExecutionBatch + Scheduler                    |
| source/WeightPack loading     | ExternalBindingPlan + backend object store    |
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
  - Add ferrule-graph
  - Add graph_runtime.rs with ExecutionBatch and ExternalBindingPlan

Phase 1: graph construction
  - Build dense decoder templates
  - Generate ComputeGraph without executing it
  - Produce ExternalBindingPlan from semantic roles
  - Add shape validation registry

Phase 2: CPU/reference graph executor
  - Interpret graph nodes through reference registries
  - Reuse existing runtime reference helpers
  - Use this for graph/dialect/backend parity

Phase 3: CUDA graph backend adapter
  - Lower graph nodes internally to CudaSourceOperatorContext
  - Add scratch allocation and memory planning
  - Add pattern fusion
  - Add optional CUDA driver graph capture

Phase 4: graph-backed RuntimeRunner
  - Add RuntimeRunner::Graph(...)
  - Use graph runner for new dense-family support first
  - Keep OLMoE and DeepSeekV4 legacy paths until parity exists

Phase 5: MoE and DeepSeek migration
  - Add routed/shared expert graph fragments
  - Validate expert residency and streaming through ExternalBindingPlan
  - Add MLA/HC/hash-routing dialect support before moving DeepSeekV4

Phase 6: training/autograd
  - Add GradientRegistry
  - Generate backward graph or tape
  - Reuse backend execute boundary
```

## Open questions

- Should dialect op-key helpers live in `ferrule-runtime`, or should there eventually be a `ferrule-dialects` crate?
- Should shape inference be mandatory during graph construction, or can some backend-specific shapes remain lazy?
- How strict should `ExternalKey` naming be? Prefer `TensorRole + layer + slot` over raw tensor names.
- Should `TensorData` support borrowed or mmap-backed payloads for CPU reference graph execution?
- Should `ferrule-cuda::graph` be renamed to `ferrule-cuda::cuda_graph` to avoid ambiguity with `ferrule-graph`?
- What is the minimal graph fragment that proves dense decoder coverage without overfitting to one family?
