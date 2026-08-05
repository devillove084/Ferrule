# Ferrule Target Architecture

<!-- markdownlint-disable MD013 MD024 MD060 -->

> Status: normative target architecture; this document does not claim that planned capabilities are implemented.
> The release plan and current evidence remain authoritative in [ROADMAP](ROADMAP.md).
> The native boundary is specified separately by [KERNEL_PROVIDER_ABI](KERNEL_PROVIDER_ABI.md); that companion document must exist before the ABI is frozen.

## 0. Purpose and scope

Ferrule is a **Rust-owned exact Mixture-of-Experts runtime**: Rust owns lifecycle, scheduling, memory, I/O, residency, transactions, cancellation, and publication while model semantics remain exact.
The current CUDA profile is the first hardware profile, not Ferrule's architectural identity; later profiles may change providers without weakening these contracts.

This document defines the target control plane and its non-negotiable boundaries; `MUST`, `MUST NOT`, `SHOULD`, and `MAY` are normative.
Model-specific algorithms may extend the data plane without acquiring runtime ownership; performance policy may change ordering but never exactness or hard feasibility.

The target decomposition is:

```text
serving/API
  -> Rust transaction owner and scheduler
  -> model-neutral execution protocol
  -> model adapter and immutable prepared plan
  -> storage provider + device provider + kernel provider
  -> hardware profile
```

## 1. Architectural invariants

### 1.1 Rust ownership

- One Rust owner is the sole writer of authoritative runtime state.
- The owner controls requests, transactions, queues, credits, waiter indices, KV branches, arenas, residency, generations, leases, and retirement.
- For the current CUDA profile, the model-owner thread is the only CUDA submitter and owns contexts, streams, events, graphs, and allocations.
- Workers receive owner-authorized immutable commands and return opaque completions; they never publish residency or mutate transactions.
- Native providers consume versioned POD descriptors containing Ferrule-owned pointers and handles.
- Providers allocate nothing, retain no hidden request state, perform no host synchronization, and do not choose fallback semantics.
- Missing provider capability, plan, exact shape, format, or target is an error.
- A future transfer thread or non-CUDA profile requires an explicit ownership profile; it is not an implicit exception.

### 1.2 Exactness and publication

- Predicted routes and prefetch hints are advisory and never become authoritative expert bindings.
- Exact dependencies are established only from model execution at a declared safe boundary.
- Every compute dispatch uses generation-validated bindings protected by leases until its completion fence.
- Provisional model state, runtime KV, backend KV, and external tokens publish atomically at commit.
- Rollback exposes no provisional output and restores the declared checkpoint exactly.
- Cancellation permanently forbids future commit for that transaction.
- No stale completion, timeout, fairness rule, or memory pressure may bypass generation validation.
- Every reservation, submission, lease, branch, frame, event, and credit has exactly one retirement path.

### 1.3 Boundedness and liveness

- All physically scarce resources are governed by hard credits before use.
- Submitted work retains its credits until a completion proves that referenced memory and handles are reusable.
- Soft policy may defer work but may not oversubscribe hard credits.
- Admitted execution receives bounded progress capacity; optional Prefetch is suppressed first.
- No owner loop may depend on busy polling, full-table rescans, or broadcast wakeups for progress.

## 2. Execution transaction

`ExecutionTransactionId` identifies one provisional execution and outlives every continuation it owns.
A transaction payload owns all model states, provisional KV/mutable generations, scheduler/session state,
continuations, materialization custody, actions, and backend identity required to finish or abort it.
At most one live transaction may publish for a request generation.

Ferrule does **not** define one giant cross-workload transaction state enum. Ordinary inference,
speculation, training, and RL have different necessary payload phases. They share only the physical
terminal contract:

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

- `Pending` means physical/backend work may still access transaction resources. The owner retains the
  entire payload and automatically calls `end_transaction()` again from a later tick/wake.
- `Complete` guarantees the backend no longer accesses transaction resources. Only then may the owner
  publish/discard provisional generations, finish materialization custody, detach continuations,
  release KV/branches, and restore scheduler/session ownership.
- `Err` is a fatal backend/protocol failure. It is reported to the worker/service; Ferrule does not
  add device-loss recovery or device-quarantine transaction states.

Workload-local phases remain minimal. Current inference uses `Executing / Publishing / Aborting` and
speculation uses `Proposing / Verifying / Ending(Publish|Abort)`. These are payload phases, not
separate terminal protocols.

### 2.1 Cancellation, failure, and publication

Cancellation records `Abort` once and forbids future publication. Repeating a business cancellation
is not a progress mechanism; owner ticks drive pending aborts automatically.

The mandatory order is:

```text
request Publish or Abort
  -> backend Pending until physically quiescent
  -> backend Complete
  -> publish committed generation OR discard provisional generation
  -> finish transaction materialization custody
  -> detach continuations and release KV/model branches
  -> restore, requeue, fail, or finish scheduler/session ownership
```

No logical cancellation, timeout, or error may release physical resources before backend
`Complete`. Cleanup failure after `Complete` is retained as a typed operational error and never
reclassifies the backend transaction as active.

Mutable state SHOULD use provisional generation plus atomic publication. Rollback SHOULD discard an
unpublished generation after quiescence rather than restore arbitrary bytes in place. Large optimizer
state MAY use shard generations, copy-on-write extents, journals, or atomic manifests.

## 3. Runtime-wide materialization

### 3.1 `LoadRegistry` identity

One runtime-wide `LoadRegistry` owns physical materialization for parameters, routed experts, KV
state, activation checkpoints, gradients, and optimizer state.
Its key includes every property that can change bytes, execution meaning, placement, or completion validity:

```rust
struct LoadKey {
    model: ModelIdentity,
    source: SourceIdentity,
    source_hash: ContentHash,
    layer: LayerId,
    expert: ExpertId,
    format: ExpertFormat,
    target_backend: BackendIdentity,
    target_device: DeviceIdentity,
    destination_generation: DestinationGeneration,
}
```

`destination_generation` includes the reserved destination identity and reuse generation; keys differing in any field are never coalesced.
One exact key has at most one active physical load. This is registry-internal physical dedup, not a
caller-visible single-flight owner, waiter, continuation, or transaction abstraction. Cache residency
shares identity rules but has a distinct lifetime.

### 3.2 `LoadOp` lifecycle

The complete state machine is:

```text
Reserved -> ReadSubmitted -> HostReady -> UploadSubmitted -> Installing -> Resident -> Retired
Reserved -> Retired
ReadSubmitted | HostReady | UploadSubmitted | Installing -> Failed | Stale
Failed | Stale -> Draining | Retired
ReadSubmitted | UploadSubmitted | Installing -> Draining
Draining -> Retired
```

The Rust owner is authoritative in every state; a worker/provider has only temporary physical custody named by an owner command.

| State | Custody and credits | Last waiter, stale handling, and exit |
|---|---|---|
| `Reserved` | Owner holds unsubmitted SQE/slab/read/upload/frame grants. | Last waiter releases immediately; stale destination retires this op and creates a newly keyed op. |
| `ReadSubmitted` | Storage command has custody; SQE, aligned slab, read bytes, and destination reservation remain charged. | Last waiter requests cancellation but waits for CQE; stale completion enters `Stale`. |
| `HostReady` | Owner has immutable host artifact; read/SQE may retire, while slab address, upload, and frame grants remain. | Last waiter may discard before upload; destination mismatch enters `Stale` and rebinding uses a new op. |
| `UploadSubmitted` | Device command has custody; slab, upload/event, and frame grants remain through its fence. | Last waiter marks an orphan but cannot preempt; stale event enters `Stale`. |
| `Installing` | Owner validates event, source, destination, and generation while publication grants remain held. | Last waiter may suppress publication; mismatch enters `Stale`; success alone enters `Resident`. |
| `Resident` | Owner publishes; frame/lease responsibility transfers atomically to the residency record. | Exact waiters are targeted-woken; the load op retires only after transfer is recorded once. |
| `Failed` | Owner stores typed failure; submitted references remain charged. | Waiters fail, never become ready; retry is a new op; drain if any command is outstanding. |
| `Stale` | Owner forbids publication and retains referenced credits. | Live exact waiters atomically detach/rebind to a fresh key/op; this op drains or retires. |
| `Draining` | Owner ledger remains authoritative while command custody and all unsafe-to-release credits persist. | No waiter is required; each CQE/event closes one obligation and the last closes to `Retired`. |
| `Retired` | Owner tombstone has zero grants, custody, and open obligations. | Duplicate completion/detach is diagnosed and cannot rebind, republish, or retire twice. |

Each op has one monotonic retirement record keyed by `LoadId`; only the owner closes it.
Stale rebinding never mutates a stale key or recycles its destination generation, and exactly-once retirement is required on success, failure, cancellation, and orphan completion.

### 3.3 Prefetch ownership

Prefetch is an explicit optional owner, not a synthetic execution dependency:

- `PrefetchOwner` distinguishes model warmup, transaction-scoped exact phase ownership, and external
  callers; their identity namespaces cannot collide;
- `ResourceDemand` is either `ModelWarmup`, `Prefetch(ExecutionPhaseSet)`, or
  `Required(ExecutionPhaseSet)`; it preserves exact phases and expresses only whether work blocks
  physical execution;
- the registry never preempts submitted physical custody, while its workload owner controls only new
  model-warmup admission: startup submits a small physical wave and opens request admission;
  foreground drive reserves no new warmup operation but drains every admitted read/upload/install;
  idle drive fills the planned residency high-water at full physical throughput;
- a model-derived Required-wave execution reserve and hard capacity remain available to exact misses;
  this is one registry/provider path, not full-fit/partial-fit executors or a second admission mode;
- the model residency planner bounds warmup at its VRAM high-water mark: full-fit eventually declares
  and installs every immutable resource, while partial-fit declares only the planned slot capacity and
  cannot create an unbounded eviction loop;
- full-fit does not use predictive expert prefetch: idle warmup establishes steady-state residency and
  a cold foreground request promotes only exact identities; a partial-fit predictor is production-ready
  only after demonstrating positive end-to-end benefit including displacement and H2D contention;
- Prefetch creates no waiter, continuation, materialization custody, execution lease, or hard progress
  entitlement; transaction ownership already exists independently;
- Required execution may join an active Prefetch only for the same exact key/generation/slot and may
  acquire the provider execution lease only after hard admission;
- provider `prepared(key)` is observation-only and MUST NOT promote purpose;
- whenever owners change, operation demand is recomputed from all remaining Prefetch owners and
  Required waiters; queued work is reclassified without changing submitted physical custody;
- multi-key execution admission is failure-atomic: acquired execution leases are compensated, newly
  created work is reclaimed, and original Prefetch ownership survives;
- transaction-scoped Prefetch is released automatically only after backend terminalization completes;
- cancelling the last Prefetch owner follows the same physical custody rule as any other submitted
  operation: submitted work drains before resources are reusable.

### 3.4 Waiters and wakeup

The registry maintains both indices:

```text
load_to_waiters: LoadId   -> Set<WaiterId>
waiter_to_loads: WaiterId -> Set<LoadId>
```

A `WaiterId` contains transaction, request generation, dependency-set epoch, and continuation identity; owner-thread attach/detach updates both indices atomically.
A waiter becomes runnable only when every exact dependency is published and generation-valid; completion wakes only affected waiters.
Duplicate completion/detach is detected and cannot double-release resources.

Cancellation detaches through `waiter_to_loads`; shared work continues while other waiters remain, and an unsubmitted last-waiter operation retires immediately.
A submitted last-waiter operation becomes orphaned or cancellation-requested and drains to its CQE/event before reuse.

Read/upload failure marks the load failed, retains credits through safe retirement, and sends attached waiters a typed transition to `Rollback` or `Cancelling`, never readiness.
A stale completion never publishes or satisfies a dependency; the owner drains it and re-resolves live waiters against a new generation or returns typed failure.

## 4. Hard credits

The global owner-written ledger records non-cloneable grants; admission reserves every credit needed by the next transition atomically and rolls back partial reservation deterministically.
Cross-thread resource lifetime follows sole-writer command/return, not arbitrary `Drop`:

```text
owner -> worker/provider: Command { operation_id, grant_ids, immutable descriptors }
worker/provider -> owner: Completion { operation_id, stage, result }
```

Only the owner mutates the ledger or releases a submitted stage.
Owner-local `Drop` may roll back an unsubmitted grant; worker/provider drop, channel close, panic, or timeout only creates an owner-visible obligation and cannot return credit implicitly.
Every profile defines finite capacities for at least:

| Credit | Protects |
|---|---|
| `sqe` | submitted `io_uring` operations and matching CQE obligations |
| `slab` | registered host slabs, descriptors, and pinned bytes |
| `read_bytes` | aligned physical bytes in flight or retained for completion |
| `upload` | placement slots, copy bytes, and device events |
| `frame` | destination frames/slots and mapping generations |
| `lease` | bindings that cannot be evicted or republished |
| `arena` | activation, workspace, and packed execution arenas |
| `kv` | provisional pages, page-table entries, and branch metadata |
| `continuation` | parked transaction/continuation objects and retained state |
| `ready_cohort` | isolated runnable packed cohorts awaiting or holding dispatch capacity |

Aggregate host/device/UMA bytes are an additional hard envelope, not a substitute for structural credits; grants transfer with ownership and retire by safe stages exactly once.
Cancellation cannot refund issued reads or release referenced memory, and no emergency lane, overflow, deadline, or fairness promotion may cross a hard bound.

An `execution_reserve` protects continuation, ready-cohort, arena/KV, and physical I/O capacity from
optional Prefetch. Only `ResourceDemand::Required` may use it; `ModelWarmup` and `Prefetch` may not.
After command submission a grant is non-preemptible until CQE/event/fence return, and reserve is
restored by blocking later optional work rather than stealing credits. Optional work has no hard
progress entitlement and is suppressed first.

## 5. Scheduling, fairness, and critical path

The scheduler is work-conserving among exact, dependency-complete, hard-feasible transitions and never lets I/O-blocked work occupy a runnable cohort slot.
Selection is lexicographic: hard feasibility, cancellation/deadline obligations, runnable work, fairness, marginal physical cost, phase policy, deterministic tie-break.

Materialization fairness has exactly three private policy bands: `Background`, `Prefetch`, and
`Required`. These bands do not replace `ExecutionPhaseSet` or become transaction states: exact
prefill, decode, proposal, verification, multimodal, training, optimizer, rollout, and reward phases
remain attached to `ResourceDemand`. Foreground ownership suppresses only new `ModelWarmup` reserve
actions before fairness selection; it does not block completions or the remaining stages of admitted
physical work. Fairness combines bounded aging with deficit round robin (DRR):

- each band has a configured positive `quantum`, finite surplus cap, and finite debt limit;
- every transition declares an exact charge no greater than `max_transition_cost`; larger work is split or rejected before admission;
- each round adds `quantum`; dispatch subtracts actual marginal cost, and an aged exact transition may enter debt only down to the configured limit;
- configuration must let one minimum atomic exact transition fit both `max_transition_cost` and all hard credits;
- joining an existing exact-key operation has zero duplicate physical-byte charge;
- required waiting age rises to a hard starvation ceiling; execution-phase policy orders runnable work
  without erasing the exact phase set;
- DRR debt changes ordering only, never hard feasibility, and speculative prefetch earns no mandatory service.

Every transaction/shared load contributes timestamped DAG nodes for proposal, route, read, host-ready, upload, event, publish, compute, acceptance, KV publication, rollback, and retirement.
Edges encode data, stream/event, credit, generation, and commit order; the measured critical path is the longest causal path to external commit or retirement.

`uncovered wait` is dependency stall left on the critical path after useful independent execution and legal overlap, not total read latency, summed waiter latency, queue residence, or assumed `read + upload`.
Telemetry separates service, credit, policy, covered, and uncovered wait; overlap claims require DAG evidence of independent work during another transaction's wait.

## 6. Typed error boundary

Runtime orchestration uses one SNAFU `Error` boundary. Backend, I/O protocol, materialization,
registry, fairness, and physical-resource failures remain typed `source` variants. A primary failure
plus one or more cleanup failures is represented as `Cleanup`, `CleanupBatch`, or `CleanupFailures`;
internal code MUST NOT flatten these sources into strings.

`InvalidRequest` is reserved for invalid caller/configuration input and `Invariant` for runtime-owned
state violations. `Pending` is never encoded as an error. String formatting is allowed only at a true
process/API/IPC boundary where the typed value cannot cross.

## 7. Model-neutral execution boundary

The runtime-facing poll contract is model-neutral:

```rust
enum Progress<T> {
    Runnable(T),
    Waiting(DependencySet),
    Complete(ExecutionResult),
}
```

`Progress::Waiting(DependencySet)` is the only model-to-runtime external wait; dependencies are typed opaque IDs/generations, not DeepSeek, CUDA, file, or request-internal objects.
Adapters expose exact dependencies only at safe yield boundaries and cannot poll storage, own event loops, or retain unreported credits.

`ExpertLeaseSet` is the exact dispatch authority for one segment:

```rust
struct ExpertLeaseSet {
    bindings: Box<[GenerationValidatedBinding]>,
    mapping_epoch: MappingEpoch,
    completion_contract: FenceContract,
}
```

The owner revalidates and atomically acquires the whole set before dispatch; after submission it is immutable through the completion fence.
Partial acquisition is abandoned, and model code sees semantic bindings rather than storage extents or upload mechanics.

Storage and device are separate injected capabilities:

```rust
trait ExpertStorageBackend {
    fn submit_read(
        &mut self,
        source: SourceExtent,
        slab: &RegisteredPinnedAlignedSlabGrant,
        read: &ReadCreditGrant,
    ) -> Result<ReadTicket>;
}
```

The current CUDA-profile `O_DIRECT` implementation reads immutable extents directly into the registered, pinned, alignment-valid slab: no pageable bounce buffer, hidden reallocation, or intermediate host copy is permitted.
Source offset, length, and address alignment are validated before SQE submission; short/partial completion cannot expose `HostReady` until the exact extent is complete.
The owner retains the slab grant and stable address from read submission through `HostReady`, transfers that obligation into the upload command, and releases it only after the upload fence.
This is storage-side zero-copy into staging followed by one explicit placement, not a claim of NVMe-to-GPU GDS.

```text
ExpertDevice:  HostReadyArtifact + frame/upload grants -> placement event
KernelProvider: exact plan + buffers + ExpertLeaseSet -> compute event/result
```

Storage cannot call device providers and device code cannot reopen model files; capability negotiation is explicit and versioned by `KERNEL_PROVIDER_ABI`.

## 8. Testing without a GPU

The complete owner state machine runs with deterministic fake providers and no GPU: `FakeStorage` controls reads/CQEs/races/errors, while `FakeDevice` controls generations/events/failures/fences.
A virtual clock drives events without sleeps or polling.
Tests exercise the production `LoadRegistry`, waiter indices, credits, transaction transitions, and scheduler policy.

Required invariant tests include:

- N waiters for one key issue one physical read and receive targeted wakeups;
- cancellation of one waiter preserves shared work, while last-waiter cancellation drains safely;
- read failure, upload failure, and stale generation never publish or report readiness;
- all credits remain bounded and return to baseline after every terminal path;
- decode reserve makes bounded progress under sustained prefill and prefetch pressure;
- aging/DRR prevents starvation with unequal expert byte costs;
- commit, rollback, cancellation, and failure reconcile KV and external tokens exactly;
- DAG accounting distinguishes covered from uncovered wait;
- replay with the same event trace produces the same decisions and state hashes.

GPU integration tests then validate ABI descriptors, event ordering, numerical parity, and real fence lifetime; they do not replace fake-backend state-machine coverage.

## 9. Current mechanism disposition

This section is migration guidance, not a claim that target replacements exist.

### Preserve

- the single Rust model-owner authority and current-profile single CUDA-submitter rule;
- stable `ExecutionTransactionId`, provisional paged-KV commit/rollback, and external-token reconciliation;
- stable expert slot/generation/lease semantics and failure-atomic publication;
- `O_DIRECT + io_uring` registered-slab reads as the first current-profile storage path;
- exact pre-MoE continuation, owned arenas, completion reactor, and deferred sequence/KV cleanup foundations;
- deterministic queue rotation, advisor traces, and existing hard-resource broker tests.

### Replace incrementally

- model-local/per-layer in-flight load ownership with the global keyed `LoadRegistry`;
- batch-local union estimates as physical truth with registry-derived exact-key dedup accounting;
- model-specific `Waiting` payloads and polling with `Progress::Waiting(DependencySet)`;
- broad completion polling/wake with bidirectional waiter indices and targeted wake;
- fragmented I/O permits with one staged hard-credit ledger covering the full transaction lifetime;
- implicit lease vectors with atomic `ExpertLeaseSet` dispatch authority;
- shared runner/arena topology with transaction-isolated multi-outstanding topology only at A6.

### Delete after replacement parity

- duplicate authoritative read/upload maps and any path that can issue a second load for the same `LoadKey`;
- whole-cycle `ResidentReady` or predicted-all-hit as proof of exact dispatch readiness;
- busy polling, full waiter scans, broadcast wakeups, and timer-only retry loops;
- soft overflow that can exceed SQE, memory, frame, lease, arena, KV, continuation, or cohort credits;
- cleanup that releases submitted resources before CQE/event/fence retirement;
- the proposal serialization guard only after the A6 topology gate passes.

Today the proposal guard intentionally serializes proposal-enabled continuations to protect shared topology and remains mandatory through A0–A5.
Only A6 may permit multiple outstanding proposal transactions after its isolation and failure gates pass.

## 10. Migration plan and exit gates

### A0 — Freeze behavior and observability
Unify stable state/load/operation IDs, trace schema, and a legacy obligation ledger around current mechanisms without changing admission or requiring complete credit grants.
Keep the proposal serialization guard and enumerate capacity/ownership gaps for A2.
**Exit gate:** current suites pass; every submitted read/upload/compute opens one owner/generation obligation and each completion closes it at most once; uncredited legacy resources are explicitly inventoried rather than claimed covered.

### A1 — Introduce neutral contracts and fake providers
Add `Progress`, `DependencySet`, `ExpertLeaseSet`, and separate storage/device/kernel traits behind adapters to current implementations.
Build deterministic fake storage/device providers and virtual-clock event driving.
**Exit gate:** transaction wait/resume, commit/rollback, cancellation, stale event, and failure scenarios pass without CUDA; production behavior remains parity-clean behind adapters.

### A2 — Unify hard credits
Replace the A0 legacy obligation ledger with complete owner-issued grants for SQE, slab, read-byte, upload, frame, lease, arena, KV, continuation, and ready-cohort admission.
Add aggregate profile memory accounting, command/return stage release, decode reserve, and retirement audits.
**Exit gate:** every physical and structural use requires a grant; pressure/failure never exceeds capacity or leaks credit, submitted grants are never revoked, and terminal gauges return to baseline; the proposal guard remains enabled.

### A3 — Land global `LoadRegistry`
Route exact misses through the full `LoadKey`, exact-key physical dedup lifecycle, bidirectional waiter indices, and targeted wake.
Adapt existing current-profile reads/uploads while retaining storage/device separation.
**Exit gate:** fake and integration tests prove one physical load per key, exact join accounting, safe cancellation, typed failure, stale-generation rejection, and no unrelated wakeups.

### A4 — Make transactions fully explicit
Move branch, checkpoint, arena, continuation, dependencies, leases, cancellation, quiescence, and retirement under the normative transaction state machine.
Remove hidden resources from model continuations and emit the causal DAG.
**Exit gate:** every legal transition is covered, every illegal transition fails closed, commit/rollback remains atomic, and long cancellation/failure runs retire exactly once; proposal execution is still serialized.

### A5 — Activate bounded scheduling policy
Schedule only dependency-complete, hard-feasible transitions; add aging, byte-cost DRR, decode reserve, ready-cohort accounting, and critical-path/uncovered-wait metrics.
Exercise multi-transaction policy in simulation and on topology-safe non-proposal work, not by bypassing the proposal guard.
**Exit gate:** deterministic traces prove work conservation, bounded decode progress, finite prefill progress, no prefetch entitlement, correct reserve behavior, and truthful covered/uncovered wait accounting.

### A6 — Isolate and enable multi-outstanding topology
Give each proposal transaction isolated runner state, packed metadata, arena checkout, branch/KV state, continuation identity, wake epoch, lease set, and cleanup path.
Allow independent ready cohorts to run while another transaction waits for read/upload completion.
Remove the serialization guard only after all prior gates pass.
**Exit gate:** multiple proposal-enabled transactions survive mixed full/partial/zero acceptance, cancellation, read/upload/compute failure, stale completion, and memory pressure with exact parity and bounded credits; traces show targeted wake and measured cross-cohort overlap; disabling multi-outstanding restores the A5 behavior without changing semantics.

## 11. Final architecture gate

The target is reached only when A0–A6 pass on fake providers and the current CUDA profile; useful overlap additionally requires a critical-path DAG showing reduced uncovered wait on production-shaped traces.
Current CUDA-profile assumptions remain profile-specific, scaffolding is not implementation, and release claims/evidence remain governed by [ROADMAP](ROADMAP.md).
