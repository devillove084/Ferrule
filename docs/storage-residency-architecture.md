# Ferrule Storage and Expert Residency Architecture

_Status: canonical target design; current implementation is partial_

_Last updated: 2026-07-13_

This document defines storage/residency ownership after the execution-engine
refactor. It is intentionally narrower than the previous survey: it describes
Ferrule's concrete lifecycle, current mechanisms, target coordinator, interfaces,
metrics, and staged migration.

Related documents:

- [`execution-engine-architecture.md`](execution-engine-architecture.md) —
  prepared executor, sequence state, physical KV, device router, and E6
  integration.
- [`ROADMAP.md`](ROADMAP.md) — E0–E8 dependencies and gates.
- [`status/2026-07-11-gb10-dsv4.md`](status/2026-07-11-gb10-dsv4.md) — current
  DSV4 measurements and known hot-path costs.

---

## 1. Scope and decisions

Ferrule targets sparse MoE inference under memory pressure. DSV4 routed experts are
the immediate pressure test, but the vocabulary must remain model-neutral.

The target separation is:

```text
model semantics
  router policy, selected-expert demand, artifact layout, optional prediction hint
        │
        ▼
runtime ExpertResidencyCoordinator
  budget, hotness, retain/prefetch/evict, sequence/batch admission, leases
        │ ResidencyPlan / transfer intent
        ▼
CUDA backend residency store
  resident slots, staged/pinned sources, upload stream/events, generations
        │ stable expert indirection
        ▼
prepared MoE dispatcher / grouped kernels
```

Decisions:

1. Runtime owns **cross-request policy**, not model-local runner state.
2. CUDA owns **physical expert handles, transfers, events, and indirection**, not
   scheduling fairness or request identity.
3. Model provides **demand and semantics**, not transfer orchestration.
4. Residency must not change routing results, weights, token order, or numerical
   reduction order.
5. Selected experts hold leases through dispatch; they cannot be evicted while a
   batch/graph binding references them.
6. A stable expert slot plus generation is required before long-lived CUDA graphs.
7. Out-of-core residency and distributed expert parallelism are separate concerns.

---

## 2. Current code reality

The prior document is outdated in several important ways. Current DSV4 CUDA code
already has:

- `HostStagedExpertCache`;
- CUDA pinned expert-source cache;
- asynchronous host staging;
- dedicated upload stream/events;
- in-flight upload tickets;
- resident FP4 CUDA handles;
- selected/prefetch classification and counters;
- model-local `ExpertStreamingPlanner` and predictor;
- deterministic route-ranked MoE reduction.

The remaining problem is ownership, not absence of mechanisms:

| Current location | Contains | Why it is not the target |
|---|---|---|
| `ferrule-model::moe::streaming` | planner, sources, host staging vocabulary | policy is still invoked by a model runner and cannot coordinate requests globally |
| `DeepSeekV4Runner` | prediction, lookahead, prefetch sequencing | one implicit sequence owns behavior that should be shared/runtime policy |
| `DeepSeekV4CudaOperatorCache` | resident handles, upload tickets, pinned cache, host cache, transfer counters, MoE workspace | backend resources are mixed with prepared weights, sequence KV, arenas, and diagnostics |
| `ferrule-runtime::expert_residency` | generic trait/vocabulary | active DSV4 CUDA path does not yet use it as the authoritative coordinator |
| `ResidentScheduler` | request admission and sequence actions | cannot budget or reserve actual expert resources before execution |

The GB10 cold path remains dominated by prompt-dependent selected expert
materialization. Stable numerical parity does not make the current design a serving
residency system.

---

## 3. Vocabulary

The existing storage vocabulary remains useful:

```text
StorageObjectId   logical identity of immutable bytes
ObjectLocator     how bytes can be fetched
Placement         where a replica is located
ObjectReplica     one concrete replica at one placement
TransferEngine    reads/copies between placements
Policy            retain / prefetch / evict decision
```

The execution engine adds the following concepts:

| Type | Meaning |
|---|---|
| `ExpertKey` | Model-instance-qualified immutable expert identity, not merely `(layer, expert)` across all models. |
| `ExpertSlot` | Stable backend-local index used by device dispatch/indirection. |
| `ExpertGeneration` | Monotonic slot/resource version; rejects stale dispatch and graph metadata. |
| `ExpertLease` | Temporary non-evictable reference acquired for an execution batch. |
| `TransferTicket` | Async staging/upload operation with source, destination, event, bytes, and result state. |
| `ExpertDemand` | Selected and predicted expert requirements emitted by model lowering. |
| `ResidencyPlan` | Runtime policy decision: retain, prefetch, acquire lease, evict, or reject. |
| `ResidencyOutcome` | Backend result: resident, in-flight, host-staged, cold, failed, bytes/wait metadata. |

`ExpertKey` must be namespaced by model instance or model fingerprint. Current
layer/expert IDs are not sufficient for a backend that can host multiple models.

---

## 4. Ownership and interfaces

### 4.1 Model-side demand

The model provides deterministic semantics:

```rust
pub struct ExpertDemand {
    pub layer: u32,
    pub selected: Vec<SelectedExpert>,
    pub predicted: Vec<PredictedExpert>,
    pub route_capacity: u32,
}

pub struct SelectedExpert {
    pub key: ExpertKey,
    pub router_rank: u32,
    pub weight: f32,
}
```

DSV4 may add hash/DSpark lookahead hints, but those hints do not directly upload or
evict expert bytes.

### 4.2 Runtime coordinator

```rust
pub trait ExpertResidencyCoordinator {
    fn plan(
        &mut self,
        batch: &ExecutionBatch,
        demand: &[ExpertDemand],
        budget: ResidencyBudget,
    ) -> Result<ResidencyPlan>;

    fn commit(&mut self, outcome: &ResidencyOutcome) -> Result<()>;

    fn release(&mut self, leases: &[ExpertLease]) -> Result<()>;
}
```

Runtime responsibilities:

- per-tier memory budget;
- hotness and workload history;
- cross-request retain/prefetch/evict policy;
- admission/deadline interaction;
- sequence/batch ownership and leases;
- metrics aggregation;
- poisoned request handling.

It does not own CUDA handles or disk/mmap implementation.

### 4.3 CUDA backend residency store

```rust
pub trait ExpertResidencyBackend {
    fn prepare(
        &mut self,
        plan: &ResidencyPlan,
    ) -> Result<ResidencyOutcome>;

    fn poll(&mut self) -> Result<Vec<TransferTicket>>;

    fn acquire_slots(
        &mut self,
        leases: &[ExpertLease],
        dispatch: &mut PreparedMoeDispatch,
    ) -> Result<()>;

    fn release_slots(&mut self, leases: &[ExpertLease]) -> Result<()>;
}
```

CUDA responsibilities:

- immutable expert bundle decode/upload;
- host-staged and pinned sources;
- upload stream/event recording;
- resident device handles;
- slot allocation and generation;
- stable device indirection table;
- event dependency insertion on compute stream;
- physical eviction after lease release;
- backend-local counters.

### 4.4 Immutable source versus replica

A source is immutable and may be shared:

```text
artifact shard / mmap byte range
host decoded bundle
pinned host bundle
managed/device expert handle
```

A replica is mutable residency state. It has placement, bytes, generation, and
optional transfer ticket. Do not encode source tier and resident state in one enum.

---

## 5. Target execution flow

### 5.1 Steady resident path

```text
model layer prefix produces router hidden
→ device router top-k/weights/grouping
→ model emits ExpertDemand
→ runtime confirms/renews leases
→ backend resolves stable ExpertSlot + generation
→ backend writes compact dispatch metadata
→ grouped FP4 kernels execute
→ route-ranked reducer combines output
→ leases release after completion event
```

Required steady-state properties:

- no full router-logit D2H;
- no per-layer expert pointer-array H2D;
- no host vector construction for routes;
- no stream-wide synchronization;
- no expert allocation/upload in a warm resident workload;
- deterministic route-rank accumulation unchanged.

### 5.2 Predicted or host-staged path

```text
prediction hint / policy demand
→ host-staged source hit or asynchronous read
→ pinned source
→ upload stream ticket + event
→ compute stream waits on only required event
→ resident slot install
→ dispatch through stable indirection
```

### 5.3 Cold selected miss

Initial policy may choose request-fatal or synchronous materialization. It must be
explicit:

```text
selected miss
→ either admission blocks before execute
→ or execute returns a named residency error and sequence is poisoned
```

Do not hide selected miss with global stream sync. A later async policy may overlap
only when a source/ticket/event contract exists.

---

## 6. Stable expert indirection

Long-lived graph replay cannot capture raw pointers from a mutable `HashMap` of
resident experts. The backend needs a stable device table:

```text
ExpertSlot → {
  generation,
  gate pointer/scales,
  up pointer/scales,
  down pointer/scales,
  format/layout metadata,
  ready event state
}
```

Dispatch metadata carries slots and expected generations. Before execution:

```text
slot generation mismatch → reject / reprepare dispatch
lease missing            → reject
ready event incomplete   → wait on that event only
```

Eviction increments/replaces generation only after all outstanding leases and graph
references are retired.

---

## 7. Relationship to batch and sequence lifecycle

Residency participates in the same prepare/execute/commit boundary as KV:

```text
scheduler forms ScheduledBatch
→ runtime reserves KV and computes residency plan
→ backend prepares sequence/KV bindings and expert slots
→ PreparedStepBinding receives page/slot/route generations
→ device execute
→ completion event
→ commit sequence cursor, KV reservation, residency observations
→ release temporary leases
```

On error before commit:

- newly allocated KV pages roll back once E5 exists;
- expert leases release;
- in-flight uploads may remain backend-global cache candidates but cannot be
  installed as selected state for the failed sequence;
- sequence is poisoned until reset/release under the initial contract.

---

## 8. Metrics

Required counters by placement and outcome:

| Category | Metrics |
|---|---|
| Demand | selected, predicted, route capacity, unique experts, reuse distance |
| Device | resident hit, slot generation mismatch, lease wait, eviction, resident bytes/slots |
| Host staged | hit/miss/eviction, bytes, decode time |
| Pinned | hit/miss/eviction, bytes, pin time |
| Transfer | read bytes/time, upload bytes/time, submitted/completed/failed/in-flight tickets, event waits |
| Policy | retain/prefetch/evict/reject decisions, accuracy, budget pressure |
| Execution | grouped columns/expert, route compact time, kernel time, reducer time |
| Graph | dispatch generation invalidation, graph fallback caused by residency |

Performance reports must distinguish:

- cold selected miss;
- host-staged selected hit;
- in-flight selected wait;
- resident selected hit;
- predictive prefetch that was never selected.

---

## 9. Migration phases

### R0 — Preserve current mechanisms and measure

- Retain model-local planner, host-staged/pinned/in-flight paths, and current CUDA
  cache.
- Add counters and failure semantics required by roadmap E0.
- Do not change routing or reduction order.

### R1 — Extract backend residency store

- Split DSV4 CUDA cache into prepared weights, backend-global expert store,
  per-sequence KV, arena, and diagnostics.
- Keep an adapter so current model-local planner can drive the extracted store.
- Add model-instance namespace to expert identity.

### R2 — Introduce runtime coordinator

- Move cross-request hotness/budget/prefetch policy to runtime.
- Model emits demand/hints; backend returns outcomes.
- Keep compatibility adapter for DSV4 planner while behavior is compared.

### R3 — Device routing and stable slots

- Device top-k/weights/grouping.
- Stable slot/generation table.
- Remove host route/pointer uploads from steady path.
- Add expert leases and event dependencies.

### R4 — Native batch and graph integration

- Group experts across real multi-sequence rows.
- Use residency generation in `PreparedStepBinding` and graph bucket key.
- Graph fallback on unsupported residency state uses the same eager pipeline.

---

## 10. Non-goals

- No distributed expert parallelism, NCCL, DeepEP, RDMA, or remote cache in the
  critical E6 path.
- No use of EP as a substitute for out-of-core expert residency.
- No model-family-specific storage API.
- No raw CUDA handle in graph IR.
- No residency change that changes routing/numerics.
- No eviction of selected/leased experts.
- No graph capture around file I/O, upload, event polling, or host policy.

---

## 11. Acceptance gates

Before declaring E6 complete:

1. Device route IDs/weights match CPU/reference semantics.
2. Different residency budgets/interleavings preserve DSV4 output.
3. Selected expert leases prevent eviction races.
4. Stale slot generations and transfer failure paths are tested.
5. Warm resident path has zero router D2H, pointer H2D, and stream-wide sync.
6. Repeated workload reduces cold bytes/latency relative to current baseline.
7. Multi-sequence grouped execution preserves route-ranked deterministic reduction.

---

## 12. External alignment

Useful design references, without copying framework surfaces:

- vLLM expert parallel deployment:
  <https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/>
- SGLang expert parallelism:
  <https://docs.sglang.ai/advanced_features/expert_parallelism.html>
- SGLang P/D disaggregation:
  <https://docs.sglang.ai/advanced_features/pd_disaggregation.html>

Ferrule's immediate differentiator remains artifact-aware, out-of-core single-node
MoE residency. Distributed expert parallelism is an extension point after the
single-node lifecycle is correct.
