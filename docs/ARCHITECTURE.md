# Ferrule architecture

<!-- markdownlint-disable MD013 MD060 -->

This document describes Ferrule's implemented runtime contracts. Model-specific
numerical rules are documented in [COMPUTATION.md](COMPUTATION.md), performance
measurement in [THROUGHPUT.md](THROUGHPUT.md), and unfinished work in
[ROADMAP.md](ROADMAP.md).

## 1. System boundary

Ferrule is a Rust-owned, resource-aware inference runtime. One authoritative
Rust owner coordinates the complete lifetime of each loaded model:

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

The owner is the sole writer of requests, transactions, queues, resource
credits, waiter indices, KV branches, residency generations, leases, and
retirement records. For CUDA execution, the same owner thread creates and uses
the context, streams, events, graphs, allocations, prepared resources, and
model runner.

Workers and native providers receive owner-authorized immutable commands. They
may hold temporary physical custody, but they do not publish residency, mutate
transactions, choose model semantics, or own caller-visible state.

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

Cancellation records `Abort` and permanently forbids publication. It does not
pretend an in-flight read, upload, kernel, or event has disappeared. The
mandatory order is:

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
compiles portable and CUTLASS implementations. Ferrule currently uses CUDA 13.2
in the validated local environment and pins the header-only CUTLASS dependency
to:

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
