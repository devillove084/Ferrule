# Ferrule Storage and Expert Residency Architecture

_Status: E6 runtime expert-residency ownership is implemented; E7 graph integration remains planned_

_Last updated: 2026-07-13_

This document defines storage/residency ownership after the execution-engine
refactor. It is intentionally narrower than the previous survey: it describes the
implemented E6 lifecycle, model-neutral control ABI, runtime coordinator, physical
CUDA store, metrics, and remaining E7 graph integration.

Related documents:

- [`execution-engine-architecture.md`](execution-engine-architecture.md) —
  prepared executor, sequence state, physical KV, device router, and E6
  integration.
- [`ROADMAP.md`](ROADMAP.md) — E0–E9 dependencies and gates.
- [`expert-memory-architecture.md`](expert-memory-architecture.md) — local
  whole-expert pageable/pinned budgets, owner-thread LRU, GB10 constraints, and
  telemetry rules.
- [`status/2026-07-11-gb10-dsv4.md`](status/2026-07-11-gb10-dsv4.md) — current
  DSV4 measurements and known hot-path costs.

---

## 1. Scope and decisions

Ferrule targets sparse MoE inference under memory pressure. DSV4 routed experts are
the immediate pressure test, but the vocabulary must remain model-neutral.

The implemented E6 separation is:

```text
model semantics
  selected-expert demand, artifact layout, optional prediction hint
        │ model-neutral ExpertResidencyControl ABI
        ▼
ferrule-runtime ExpertResidencyController
  per-layer capacity/LRU, stable slots/generations, reservations, leases, stats
        │ prepared install / exact physical binding
        ▼
DSV4 CUDA physical residency store
  immutable source catalogs, staging/pinned sources, upload tickets/events,
  resident handles and stable device tables
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

## 2. Current implementation

E6 has completed the residency ownership migration:

| Location | Implemented ownership |
|---|---|
| `ferrule-common::expert_residency` | Model-neutral `ExpertKey`, stable slot/generation bindings, leases, selected/prefetch intents, two-phase prepare/publish/cancel, and stats ABI. |
| `ferrule-runtime::expert_residency` | One stable-slot coordinator per layer, capacity/LRU policy among unleased slots, selected leases, prefetch admission, cancellation, and aggregate stats. |
| `NativeMultiSessionExecutor` | Lazy controller construction/injection before execution. The injected control stays on the runner, so clean executor/driver teardown and runner re-wrapping preserve residency ownership and state. |
| DSV4 CUDA path | Immutable source-catalog access, asynchronous host staging, pinned upload sources, upload tickets/events, resident FP4 handles, stable physical tables, and physical counters only. |
| CPU/reference path | The remaining `ExpertStreamingPlanner` and `CpuExpertHandleStore` correctness implementation; CUDA does not allocate mirrors of either. |

The old CUDA planner/backend reconciliation and duplicate logical residency ledger are
removed. Selected and prefetch uploads share the pinned asynchronous upload path;
selected compute waits by inserting only the required upload-event dependency on the
compute stream. Device publication kernels update stable tables in place without a
normal-path H2D table copy or stream-wide synchronization.

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
| `ExpertSlotId` | Stable backend-local index used by device dispatch/indirection. |
| `ExpertSlotGeneration` | Monotonic slot/resource version; rejects stale dispatch and graph metadata. |
| `ExpertLease` | Temporary non-evictable reference acquired for an execution batch. |
| `ExpertInstallIntent` | Model-qualified selected or prefetch request. |
| `PreparedExpertInstall` | Unpublished exact slot/generation reservation returned before physical transfer. |
| `ExpertInstallPrepareOutcome` | Resident grant, prepared installation, or nonfatal prefetch capacity pressure. |
| `ExpertResidencyGrant` | Published binding plus the mandatory lease for selected demand; prefetch grants are unleased. |
| `ExpertResidencyStats` | Aggregate resident/lease/install/eviction/hit/cancellation/capacity counters exposed through the common ABI. |
| Upload ticket | DSV4 CUDA asynchronous pinned upload plus completion event and retained staging/resources. |

`ExpertKey` must be namespaced by model instance or model fingerprint. Current
layer/expert IDs are not sufficient for a backend that can host multiple models.

---

## 4. Ownership and interfaces

### 4.1 Model-side demand

The model owns deterministic router IDs, weights, route rank/order, immutable source
catalogs, and optional hash/DSpark prediction hints. The CUDA lowering converts compact
selected or predicted layer/expert IDs into model-instance-qualified `ExpertKey`
values and invokes the common selected/prefetch control ABI. Model hints do not choose
a slot/generation or directly publish/evict a logical binding.

### 4.2 Model-neutral control ABI and runtime coordinator

`ferrule-common` provides the object-safe `ExpertResidencyControl` ABI. Its key
operations are:

```rust
fn binding(&self, key: ExpertKey) -> Result<Option<ExpertSlotBinding>>;
fn acquire_selected(&mut self, key: ExpertKey) -> Result<Option<ExpertResidencyGrant>>;
fn release(&mut self, lease: ExpertLease) -> Result<()>;
fn prepare_install(&mut self, intent: ExpertInstallIntent)
    -> Result<ExpertInstallPrepareOutcome>;
fn publish_install(&mut self, prepared: PreparedExpertInstall)
    -> Result<ExpertResidencyGrant>;
fn cancel_install(&mut self, prepared: PreparedExpertInstall) -> Result<()>;
fn stats(&self) -> ExpertResidencyStats;
```

`ferrule-runtime::ExpertResidencyController` implements that ABI with an independent
coordinator and configured capacity for every model layer. It owns stable logical
slots/generations, LRU choice among unleased slots, selected leases, prefetch capacity
handling, prepare/publish/cancel transactions, and aggregate/per-layer stats. It does
not own CUDA handles, pinned memory, streams, events, or artifact I/O.

`NativeMultiSessionExecutor` constructs and injects the controller lazily from the
runner's model-instance and per-layer requirements. Injection occurs once before the
first execution/feed/diagnostic path that needs it. Because the boxed control is held
by the runner, `into_runner` followed by construction of a new executor recognizes and
reuses the existing control instead of replacing it.

### 4.3 DSV4 CUDA physical residency store

DSV4 CUDA owns only physical mechanisms:

- immutable expert source catalogs and host-staged/pinned source caches;
- selected and prefetch upload tickets on the upload stream;
- upload completion events and compute-stream event dependencies;
- resident FP4 device handles and their event-guarded retirement;
- stable pointer, expert-to-slot, and generation device tables;
- physical publication/eviction kernels, grouped workspaces, and backend counters.

It does not choose logical slots/generations or allocate CPU planner/handle mirrors.
Normal publication mutates the stable device tables with kernels; it performs no H2D
table copy and no stream-wide synchronization.

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

## 5. Implemented execution flow

### 5.1 Steady resident path

```text
model layer prefix produces router hidden
→ device router top-k/weights/grouping
→ compact selected IDs become model-qualified ExpertKey values
→ runtime acquires selected ExpertLease bindings
→ device kernels resolve ExpertSlotId + ExpertSlotGeneration
→ grouped FP4 kernels execute
→ route-ranked reducer combines output
→ leases release after dispatch submission
→ physical handle retirement remains guarded by compute events
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

### 5.3 Selected miss and prefetch pressure

```text
selected miss
→ runtime prepares an exact slot/generation reservation
→ DSV4 resolves/stages and pins the immutable source
→ upload stream submits an asynchronous ticket and records an event
→ compute stream waits on that event only
→ physical publication kernel updates the stable table
→ runtime publishes the reservation and returns a selected lease
```

When selected demand needs capacity, DSV4 deterministically sorts pending non-selected
prefetches and cancels the excess reservations. A canceled in-flight ticket is moved
to the abandoned-resource list: its pinned staging and device allocations remain alive
until its completion event reports that release is safe. No host event synchronize or
global stream synchronize is used to satisfy selected compute.

---

## 6. Stable expert indirection

Long-lived graph replay cannot capture raw pointers from a mutable `HashMap` of
resident experts. The backend needs a stable device table:

```text
ExpertSlotId → {
  ExpertSlotGeneration,
  gate pointer/scales,
  up pointer/scales,
  down pointer/scales,
  expert-to-slot and slot-generation metadata
}

resident handle → upload guard / retirement event
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
→ runtime reserves KV
→ runtime residency control acquires or prepares exact expert bindings
→ DSV4 CUDA completes physical staging/upload/publication
→ PreparedStepBinding receives page/slot/route generations
→ device execute
→ commit sequence cursor and KV reservation
→ release temporary leases; retire physical resources by events
```

On error before commit:

- newly allocated KV pages roll back;
- expert leases release;
- unpublished expert reservations are canceled;
- canceled in-flight uploads retain their physical resources until their events
  complete, but cannot publish the canceled logical binding;
- the sequence follows the existing transaction poison/reset rules when execution may
  have mutated state.

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
| Runtime control | resident slots, active leases, installs, evictions, resident hits, stale releases, prepare cancellations, prefetch capacity misses |
| Execution | grouped columns/expert, route compact time, kernel time, reducer time |
| Graph | dispatch generation invalidation, graph fallback caused by residency |

The aggregate runtime-controller values above are exposed through DSV4 operator
runtime counters alongside physical transfer and MoE counters.

Performance reports must distinguish:

- cold selected miss;
- host-staged selected hit;
- in-flight selected wait;
- resident selected hit;
- predictive prefetch that was never selected.

---

## 9. Migration phases

### R0/R1 — Mechanism preservation and physical-store extraction — complete

- [x] Preserve staged/pinned/asynchronous transfer mechanisms and deterministic
  routing/reduction while splitting physical CUDA resources from logical policy.
- [x] Add model-instance-qualified expert identity and separate prepared, sequence,
  arena, physical residency, and diagnostics lifetimes.

### R2 — Runtime coordinator — complete

- [x] Add the model-neutral stable slot/generation/lease and stats ABI with
  failure-atomic two-phase publication/cancellation.
- [x] Install one runtime coordinator per layer through lazy executor-to-runner
  injection, preserving it across clean runner moves.
- [x] Remove the CUDA planner/backend reconciliation and CPU planner/handle mirrors.

### R3 — Device routing, transfers, and stable publication — complete

- [x] Device score/hash routing, normalized weights, stable-slot resolution, and
  packed fixed-eight grouping validate cumulative route completion across residency
  windows.
- [x] Selected/prefetch uploads use pinned asynchronous tickets; selected compute
  waits on the upload event without host synchronization.
- [x] Device kernels publish and evict stable table bindings without H2D table copies
  or stream-wide synchronization.
- [x] Selected leases protect physical use; deterministic prefetch cancellation keeps
  in-flight resources alive until completion events retire them.

### R4 — Native batch complete; graph integration remains E7

- [x] Group real multi-sequence rows and preserve exact packed/ragged/mixed output.
- [ ] Use the completed stable residency metadata in long-lived E7 graph buckets.
- [ ] Graph fallback on unsupported residency state uses the same eager pipeline.

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

## 11. E6 acceptance validation

E6 is complete with the following implemented validation:

1. `ferrule-common`: 36 tests passed.
2. `ferrule-model`: 175 tests passed.
3. `ferrule-runtime`: 253 tests passed.
4. `just test-cuda-required` passed.
5. CUDA `expert_slot_resolve`: 5 tests passed.
6. Real DSV4 packed batch-2/batch-4 output is exact.
7. Real DSV4 ragged/mixed output is exact.
8. Real DSV4 prefix-fork COW is exact.
9. Repeated sequence execution reuses expert residency.
10. The latest 43-layer resident runtime-driver path passes.
11. Explicit device score/hash router and paged-decode CUDA gates pass with required
    zero-copy/zero-stream-sync wrapper invariants.

These gates cover stable-slot resolution, prepare/publish/cancel atomicity,
lease-protected eviction, transfer failure, asynchronous selected-event waits, native
multi-sequence grouping, and residency reuse.

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
MoE residency. The single-node E6 ownership lifecycle is implemented; distributed
expert parallelism remains a separate extension after E7 graph integration.
