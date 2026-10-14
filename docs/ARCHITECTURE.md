# Ferrule architecture

<!-- markdownlint-disable MD013 MD060 -->

This document describes Ferrule's implemented runtime contracts. Model-specific
numerical rules are documented in [COMPUTATION.md](COMPUTATION.md), performance
measurement in [THROUGHPUT.md](THROUGHPUT.md), and unfinished work in
[ROADMAP.md](ROADMAP.md). Source-pinned results and exclusions belong in
[CURRENT_ACCEPTANCE.md](CURRENT_ACCEPTANCE.md), not in this contract description.

## 1. System boundary

Ferrule is a Rust-owned, resource-aware inference runtime. A coordinator owns
logical scheduling and publication for each execution scope; physical resources
stay with the executor/worker that owns them. The resident request path is:

```text
HTTP or CLI request
  -> bounded command channel
  -> model-owner thread
       -> scheduler and execution transactions
       -> model adapter and resolved semantic stages
       -> materialization and residency
       -> paged KV and expert caches
       -> CUDA provider and completion reactor
  -> bounded result channel
  -> caller-visible token publication
```

Each mutable request, transaction, queue, credit, KV branch and residency
record has one authority. In a resident execution this is centered on the model
owner; a pipeline uses the existing coordinator and owner-local stage journals.
CUDA contexts, streams, events, allocations and prepared model resources remain
with their physical owners. This is not one thread owning every device in a
parallel execution.

Workers carry out authorized stage/expert work and retain their local state and
custody until quiescence is proven. They do not acquire the parent coordinator's
publication authority. Expert owners do not vote as KV owners. Native providers
borrow authorized descriptors/resources; they do not choose model families,
schedule requests or publish caller-visible state.

### Composition and durable layer boundaries

| Layer | Authority | Not its responsibility |
|---|---|---|
| CLI/server | Parse explicit intent, project effective configuration and HTTP/session results | Model-family lowering, a replacement inference loop or cleanup proof |
| Composition | One built-in catalog; family/profile selection, immutable resolved construction, admission and owner assembly | Forward math, a second catalog/coordinator or KV publication authority |
| Model | Checkpoint semantics, typed precision/lowering, geometry, state and operator requirements | Process lifetime, request scheduling or global commit |
| Runtime | Resident/pipeline execution, scheduling, transaction/KV publication, worker transport and custody | Choosing concrete model families or rediscovering constructors from arithmetic |
| Backend/providers | Physical buffers, kernels, streams/events and completion evidence | Serving policy or logical publication |

Catalog selection normalizes explicit request intent and returns a resolved
strategy containing construction, backend, option policy and topology policy.
The build path consumes that decision rather than reselecting a whole factory
from PP/TP/EP degrees or a precision enum. The public engine kind is
`Resident` or `Pipeline`; model family, numeric arithmetic, storage, expert
attachment and transport are separate facts. Resident EP remains resident.

Parallel composition shares validated geometry, binding, provider budget and
assembly. TP1 segment and TP-rank providers supply typed stage/collective leaves,
not independent PP×EP and TP×EP admission/prewarm/startup algorithms. Transport
constructors supply thread/process lifetime rules only for admitted profiles;
sharing assembly does not enable process TP, numeric process PP×EP or BF16
expert transport. Family adapters retain strict checkpoint/profile capability
and source validation; metadata support is not live capacity or quality proof.

The extracted source entry map is below; numerical acceptance remains solely in
[CURRENT_ACCEPTANCE.md](CURRENT_ACCEPTANCE.md), and supported combinations in
[PARALLELISM.md](PARALLELISM.md).

| Entry | Responsibility |
|---|---|
| [Catalog](../crates/ferrule-composition/src/model_factory/mod.rs) / [resolved selection](../crates/ferrule-composition/src/model_factory/selection.rs) | One built-in registration point and frozen construction/policy decision |
| Qwen adapter: [mod](../crates/ferrule-composition/src/model_factory/adapters/qwen35/mod.rs), [binding](../crates/ferrule-composition/src/model_factory/adapters/qwen35/binding.rs), [capacity](../crates/ferrule-composition/src/model_factory/adapters/qwen35/capacity.rs), [options](../crates/ferrule-composition/src/model_factory/adapters/qwen35/options.rs) | Strict family recipe/checkpoint validation, capacity facade and consumed-option policy; lower inputs for shared providers |
| Qwen [composition](../crates/ferrule-composition/src/model_factory/adapters/qwen35/composition.rs), [EP](../crates/ferrule-composition/src/model_factory/adapters/qwen35/ep.rs), [compatibility](../crates/ferrule-composition/src/model_factory/adapters/qwen35/compatibility.rs) | Family selection/capability, thin resident EP binding, and retained public facades |
| Shared [inputs](../crates/ferrule-composition/src/model_factory/parallel/inputs.rs), [bindings](../crates/ferrule-composition/src/model_factory/parallel/bindings.rs), [plan](../crates/ferrule-composition/src/model_factory/parallel/plan.rs) | Provider inputs, already-bound resource checks and geometry; no family checkpoint re-selection |
| Shared [budget](../crates/ferrule-composition/src/model_factory/parallel/budget.rs): [common](../crates/ferrule-composition/src/model_factory/parallel/budget/common.rs), [local](../crates/ferrule-composition/src/model_factory/parallel/budget/local.rs), [resident](../crates/ferrule-composition/src/model_factory/parallel/budget/resident.rs) | Provider representations and host/device/colocation admission, not a physical reservation |
| Shared [experts](../crates/ferrule-composition/src/model_factory/parallel/experts.rs), [assembly](../crates/ferrule-composition/src/model_factory/parallel/assembly.rs), [resident](../crates/ferrule-composition/src/model_factory/parallel/resident.rs), [process](../crates/ferrule-composition/src/model_factory/parallel/process.rs), [reporting](../crates/ferrule-composition/src/model_factory/parallel/reporting.rs) | Owner preparation/Ready, assembly, resident/process execution setup and reports; runtime retains commit and custody authority |

The public Rust path `model_factory::qwen35_composition` is preserved by the
`#[path]` mount in `model_factory/mod.rs`; moving its source file is not removing
that API. Qwen-named placement/capacity facades likewise wrap shared provider
calculations rather than a duplicate family factory. Private test modules can
retain their mounted names independently of source-directory names.

### Factory API migration

Concrete factories live in `ferrule_composition::model_factory` (also available
through its crate-root public re-exports), not
`ferrule_runtime::engine::model_factory`. The old runtime factory path has no
compatibility alias. Process binding is explicitly injected by composition;
runtime does not select concrete model families. Retained public aliases are
same-type/module re-exports, not duplicate owners or implementations, and do
not promise stable `std::any::type_name` strings after source/API renaming.

### Adding a model

A new family requires three pieces, not a copied PP×EP/TP×EP factory:

1. **Model recipe and strict binder:** define semantics and state/parameter
   requirements; validate checkpoint schema, shapes, storage/precision, source
   identity and supported attachments before lowering bound resources.
2. **One catalog entry and family policy:** resolve explicit backend, numeric,
   topology and serving intent into the matching construction/option policy.
   Reject unsupported or unconsumed options; lower typed inputs into the shared
   admission, resident or pipeline providers.
3. **Compatible operators and state contracts:** reuse existing execution only
   where its operators, state/KV layout, precision and completion/custody contracts
   actually fit. Otherwise implement and validate the missing model/backend
   capability first; metadata registration is neither execution nor quality proof.

Shared process preparation does **not** grant universal numeric model support:
[the numeric process binder](../crates/ferrule-composition/src/process_binding.rs)
still accepts only strict Qwen3.5-35B-A3B FP8. Another family needs an explicit
binder/boot capability and validation; this extraction does not enable process
TP, numeric hybrid process PP×EP, or BF16/Mixed expert wire.

## 2. Execution transactions

`ExecutionTransactionId` names one provisional execution. Its payload retains
all model state, paged KV generations, continuations, materialization custody,
and scheduler ownership required to either publish or abort the work.

Ordinary and speculative execution use workload-specific phases, but both end
through the same physical protocol:

```rust
enum TransactionEndIntent {
    Publish,
    Abort,
}

enum TransactionEndProgress {
    Pending,
    Complete,
}
```

`Pending` means submitted physical work may still access transaction resources.
The owner retains the entire payload and advances terminalization on a later
wake or tick. `Complete` guarantees that the backend no longer accesses those
resources. Only then may the owner publish or discard provisional generations,
release materialization custody, detach continuations, retire KV branches, and
restore scheduler ownership.

Accepted cancellation determines the caller-visible terminal outcome. Before
irreversible physical Publish starts, cancellation selects `Abort`; it does not
revoke an already-started irreversible physical Publish or rewrite its intent.
That work must still reach its physical completion/retirement boundary, without
turning the canceled request into successful caller-visible output. Cancellation
is not a fence and does not release in-flight resources. The mandatory order is:

```text
request Publish or Abort
  -> retain all ownership while backend returns Pending
  -> observe backend Complete
  -> publish or discard provisional state
  -> release custody, continuations, KV, and scheduler ownership
```

Every reservation, submission, lease, event, branch, frame, and credit has one
retirement path. Cleanup failures remain typed operational errors rather than
changing a completed backend transaction back into an active one.

## 3. Resolved execution stages

The runtime does not infer resource needs from model names or kernel choices.
The model adapter resolves each semantic stage before admission:

```rust
struct ResolvedStageResource {
    key: MaterializationKey,
    access: ResourceAccess,
    retention: ResourceRetention,
}

struct ResolvedStage {
    dependencies: Option<DependencySet>,
    resources: Box<[ResolvedStageResource]>,
    workspace: WorkspaceClaim,
}
```

A resolved stage states:

- the exact materialized identities required for progress;
- whether each resource is read, written, or read-write;
- whether custody lasts through the stage, transaction, or persistent
  residency;
- the workspace required before dispatch;
- the canonical dependency set used when the stage cannot proceed.

Keys are canonically ordered and duplicates fail closed. A resource-free stage
does not manufacture a dependency. Admission is failure-atomic: partial leases,
credits, or new load ownership are compensated if the complete stage cannot be
admitted.

## 4. Runtime-wide materialization

One runtime-wide `LoadRegistry` owns physical materialization for immutable
parameters, routed experts, KV state, activation checkpoints, gradients, and
optimizer state. `MaterializationKey` includes every property that can change
the bytes, interpretation, placement, or completion validity:

- model instance;
- source identity and content hash;
- semantic resource identity;
- payload encoding;
- backend and device;
- source generation;
- destination generation.

Keys differing in any field never coalesce. At most one active physical load
exists for one exact key. This is global physical deduplication, not a second
caller-visible transaction abstraction.

A load moves through reserved, read, host-ready, upload, install, resident,
failed or stale, draining, and retired stages. The owner remains authoritative
throughout. Submitted reads and uploads retain their slabs, byte grants,
destination reservations, and completion obligations until the matching CQE or
CUDA event proves reuse is safe. A stale completion cannot publish or satisfy a
dependency.

### Waiters and targeted wake

The registry maintains both directions of the dependency relation:

```text
operation -> exact waiter set
waiter    -> exact operation set
```

A waiter includes transaction, request generation, dependency epoch, and
continuation identity. Attach and detach update both indices atomically on the
owner thread. Completion wakes only affected waiters, and a waiter becomes
runnable only after all exact dependencies are resident and generation-valid.
No progress path depends on full-table scans or broadcast wakeups.

### Prefetch

Prefetch is optional ownership, never an exact execution dependency. Model
warmup, transaction-scoped prefetch, and external prefetch have distinct owner
identities. Required execution may join an active prefetch only for the same
exact key, destination generation, and placement.

Demand is recomputed whenever owners change. Optional work can be suppressed
before admission, but submitted work is non-preemptible and must drain. Required
execution has a protected reserve of physical and continuation capacity;
prefetch may not consume it.

## 5. Hard physical resources

`PhysicalResourceBroker` is the sole production ledger for physically scarce
resources. Admission atomically reserves every credit required by the next
transition and rolls back an incomplete reservation deterministically.

The broker bounds, as applicable:

- storage submission entries and registered pinned slabs;
- read and upload bytes;
- upload slots, CUDA events, and destination frames;
- generation-qualified execution leases;
- activation and provider workspace;
- paged KV capacity;
- continuations and runnable cohorts;
- transaction-scoped execution ownership.

Aggregate host and device bytes are additional hard envelopes, not substitutes
for structural credits. Owner-local rollback can release an unsubmitted grant.
After submission, only an owner-observed CQE, event, or fence can close the
obligation. Worker exit, cancellation, timeout, or channel closure cannot
implicitly return physical capacity.

## 6. Scheduling and overlap

The scheduler is work-conserving among exact, dependency-complete,
hard-feasible transitions. I/O-blocked work does not occupy a runnable compute
slot. Selection considers, in order:

1. hard feasibility and terminal obligations;
2. ready execution;
3. bounded fairness and age;
4. marginal physical cost;
5. execution phase policy;
6. deterministic tie-breaking.

Materialization uses background, prefetch, and required policy bands with
bounded deficit accounting. Policy changes ordering only; it never bypasses a
hard credit, generation check, or exact dependency.

Ferrule records timestamped causal nodes for route resolution, read, upload,
publication, compute, proposal, verification, commit, rollback, and retirement.
An overlap claim is valid only when the timeline proves useful independent work
occurred during another dependency wait. Summed queue residence or concurrent
submission alone is not overlap.

## 7. Paged KV and mutable state

Paged KV capacity is admitted before use. A transaction owns provisional page
reservations, page-table changes, copy-on-write branches, and prepared commits.
Commit publishes the accepted prefix atomically; abort discards provisional
state only after backend quiescence. Packed batch slots are execution-local and
must never become persistent KV identities.

Mutable resources use generation-qualified publication. A completion must match
the reserved destination and generation before it becomes visible. Leases keep
published bindings alive through the last completion fence that references
them.

## 8. Expert residency and memory

Experts are immutable compute bundles identified independently from their
current location. The owner controls source lookup, host and pinned cache
admission, upload, stable device slots, generation tables, leases, eviction,
and publication.

Host and pinned caches are owner-thread structures with simultaneous entry and
byte limits. Their hot paths require no mutex, global allocator, full-cache
scan, or per-hit atomics. Eviction removes cache retention but does not
invalidate a reference already held by an in-flight operation.

A device slot reservation is not residency. Publication occurs only after the
upload completion has been observed and the source identity, destination, and
generation have all been revalidated. Compute receives an immutable snapshot of
exact bindings protected by leases through its completion fence.

KV budgets, expert-cache budgets, transient uploads, workspace, and device
residency are distinct capacity domains. They cannot silently borrow from each
other.

## 9. Serving ownership

Ferrule uses Axum and Tokio for HTTP transport. HTTP tasks do not lock or execute
the model driver. They submit requests through a bounded command channel to one
dedicated model-owner thread and receive token events through a bounded
per-request channel.

A full or closed result channel sets cooperative cancellation instead of
blocking the model loop. The owner observes cancellation between model steps
and starts transaction abort. An already submitted CUDA batch is not
interrupted; its physical resources remain owned until completion.

The request queue is bounded and overload is rejected. Streaming and
non-streaming OpenAI-compatible completion endpoints share the same model
ownership. The implemented DeepSeek-V4 path accepts deterministic greedy
semantics and rejects unsupported sampling rather than silently changing it.

### Observability and configuration

`ferrule-common::observability::Metrics` is a process-local cumulative
accumulator; runtime/driver snapshots are separate owner-local observations
and must not be repeatedly added to process counters. The worker active owner
holds a non-cloneable `RequestMetricsGuard`, created only after successful
`engine.try_submit`. Removal/drop ends that observation exactly once. Rejected
or queued requests do not start it; cancellation acceptance or failure delivery
alone does not finish a retained owner. Do not also increment finished counters
in HTTP/terminal paths. Finished means an admitted-generation observation ended,
not successful generation, HTTP delivery, permit release, or physical cleanup.
Unknown cleanup retains its existing ownership obligations independently.
Token/latency/cache/GPU counters require explicit owner-defined observations.

`FERRULE_LOG` selects the tracing filter (default `info`), and
`FERRULE_LOG_FORMAT=json` selects JSON on stderr. Initialization is idempotent
and retains an embedded subscriber. `FERRULE_METRICS_INTERVAL` is a minimum dump
interval in seconds; missing, invalid or zero disables it. Dumping requires
explicit polling, does not reset counters, and creates no exporter, background
thread or endpoint. `shutdown()` remains a compatibility no-op.

[agent.toml](../configs/agent.toml), [train.toml](../configs/train.toml), and
[rollout.toml](../configs/rollout.toml) are legacy, non-executable TOML examples.
The CLI has no corresponding commands, `--config` option or TOML loader;
fields such as `metrics_bind` do not configure a live service. Serving CLI
runtime-limit overrides apply before owner Ready; omission preserves defaults
and unsupported engines reject explicit overrides. `/admission` is a read-only
owner-local admission/capacity sample with separately sampled HTTP leases, not
a globally atomic snapshot, cumulative metrics update or quiescence proof.
Unsupported physical capacity remains unknown/null.

## 10. Kernel provider boundary

Model and runtime code create provider-neutral semantic plans. CUDA, CUTLASS,
CuTe, architecture-specific schedules, launch geometry, and internal scratch
layouts are implementation details of `ferrule-backend`.

The native boundary is an internal same-checkout C POD ABI. It is statically
linked and does not promise compatibility across commits or releases. Any
layout change updates Rust declarations, C/C++ declarations, layout assertions,
tests, and call sites in one revision.

Rust owns:

- CUDA contexts, streams, events, graphs, allocations, and workspace storage;
- model plans, transaction state, KV, residency, and completion tracking;
- device-pointer lifetime through asynchronous completion;
- provider selection and typed error propagation.

Native code borrows the supplied pointers, handles, descriptors, stream, and
workspace. It may not allocate caller-visible or hidden device memory, retain
request state, synchronize the host, reopen model artifacts, or choose a
semantic fallback. Provider handles and borrowed views are destroyed only after
Rust has stopped new calls and observed all relevant completion fences.

Provider discovery and plan preparation validate the exact operation, target,
dtype, layout, shape, workspace, determinism, and capability set. Missing or
contradictory capability records fail closed. The hot path does not inspect
architecture strings, environment variables, or dynamic traits to choose a
kernel.

## 11. CUDA build profile

The build detects the local GPU compute capability unless
`FERRULE_CUDA_ARCH` explicitly requests a cross-build. Standard Cargo plus NVCC
compiles core and CUTLASS implementations. Native CUDA code is physically split
into two flat vertical slices, `native/cuda/core` and `native/cuda/cutlass`.
Each slice owns one aggregate `provider.cu` and its local ABI, bindings, target,
validation, and operator headers. The build compiles exactly those two
translation units independently without relocatable device code.

Ferrule currently uses CUDA 13.2 in the validated local environment and pins the
header-only CUTLASS dependency to:

```text
tag     v4.6.1
commit  e05f953a5b3d38adc240df2ff928e0421c2abba3
```

Build for the detected device:

```bash
just cuda-info
just cutlass-setup
just build-cuda
```

Compile another profile without loading it on the local GPU:

```bash
just check-cuda-arch sm_103
```

Compilation proves source and target compatibility, not numerical or lifecycle
correctness on that device.

## 12. Validation model

The owner state machine is testable without a GPU through deterministic fake
storage and device providers. Tests cover exact-key deduplication, waiter
attachment, stale generations, cancellation, credit bounds, publication,
rollback, terminalization, and deterministic scheduling.

GPU tests add ABI layout, launch, event ordering, numerical, near-tie, paged-KV,
and real fence-lifetime evidence. A test on one CUDA profile does not validate a
different architecture. Full-model token parity requires the exact checkpoint,
workload, build revision, and target hardware.
