# Runtime CPU-reference hybrid owner contract

This directory exercises the real runtime owner path. It does not replace the
parent `PipelineParallelExecutor`, `DistributedTransaction`, `KvPageManager`,
`PipelineStageWorker`, or `ExpertParallelExecutor` with a test coordinator.

## Entry point

```rust,ignore
PipelineParallelExecutor::new_hybrid_cpu_tensor(
    topology: ValidatedParallelTopology,
    spec: &DecoderModelSpec,
    layout: HybridExpertTopology,
    config: PipelineConfig,
    options: HybridCpuTensorConfig,
    load: F, // F: FnOnce(PipelineRank) -> Result<BoundDecoderResources> + Clone + Send + 'static
) -> Result<PipelineParallelExecutor>
```

The factory is CPU-reference only and is exported under the `cpu-reference`
feature from `parallel::pipeline`. It requires the authenticated PP x TP KV
mesh to match every `HybridParallelPlan::tensor_members()` vector in order.
EP is an explicit attachment, never a world-size multiplier and never a KV
participant. The factory derives `ExpertDispatchMembers` with the explicit common
`ExternalTensorSource` scope from the admitted model attachment and registers it
in `ParallelExecutionScopes`. This is the parent's sole runtime scope authority;
there is no second stored `hybrid_expert_topology`. Source authentication requires
a real TP owner of that replica/stage, including a nonzero lane. The old
`ExternalStage` scope still requires TP1, and the process decoder does not accept
the new tensor-source scope. This reference path additionally requires EP-only owners to be
disjoint from the complete KV mesh. CUDA, process transport, and legacy generic
constructors retain their existing production gates.

Each physical TP owner constructs `HybridTensorCpuSegment`, its local
`HybridRankLayout` KV planes/state/weight budget, `PipelineStage`, and the
ordinary `PipelineStageWorker` inside its persistent worker. The attached
`ExpertParallelExecutor` is created only for the explicit
`HybridExpertAttachment::source_rank`; other TP lanes do not create a replica.
`HybridTensorCpuSegment::bind_expert_call` revalidates scope, source, members and
placement before execution.

The only parent KV participants are the PP x TP mesh. EP worker ticket serials,
expert stats, and public outstanding counters are not commit or cleanup evidence.
Normal expert calls return only after correlated tickets are drained.
For TP1's registered empty wire schedule, ordinary execution error/cancellation
marks the step aborted but not its communication lifetime. Only the parent path
after ALL journal Rollback ACKs and logical retirement/cancel/retire clears that
admission. It keeps the execution watermark and does not reset any protocol or
multi-TP transport failure. Pending and Unknown retain admission/custody and
cannot be cleared by repeat shutdown/release or stats. In-flight
child work must drain before the owner returns; Unknown child completion retains
parent owner custody; Unknown reaches the
existing transport quarantine path. Owner startup and shutdown attempt cleanup
for already-created siblings and attached experts.

## Schedule and identity

```rust,ignore
translate_hybrid_schedule(
    plan: &HybridParallelPlan,
    segment: &LayerSegmentPlan,
) -> Result<Vec<TensorCollectiveSite>>
```

The only recipe is `plan.collective_schedule(segment)?.wire_sites()`; TP1 has
no wire sites. Runtime translates ordinal, global layer, semantic, kind, and
`shape { fixed, per_sequence, per_row }` without a decoder layer loop or an
agreement word count. Even the output head is semantic metadata, not a synthetic
dense linear plan. `TensorCollectiveOperator::Protocol` seals the model's
`HYBRID_AGREEMENT_VERSION`, semantic, kind and exact shape in equality. Metadata
with equal S=1 element counts but different coefficients is not equivalent.

`TensorCollectiveShape::elements(sequences, rows)` uses checked arithmetic.
`begin_sealed` preflight and exchange both use S/R from the sealed
`TensorBatchIdentity`. Owner identity is checked before backend enter; the
model's own V2 agreement exchange still compares packet contents. The parent
currently admits one sequence; it passes `batch.sequences().len()`, not a second
copy of the model's agreement recipe. The reusable decorator/collective also
supports a validated multi-sequence packed identity.

## Checkpoint and storage admission

`hybrid_tensor/admission.rs` seals the first successful owner load's payload-free
binding. Every subsequent load must match its ordered parameter specs, roles,
canonical IDs, source mappings, weight/scale slices/transforms and immutable
file snapshots before model preparation or EP startup. Each snapshot is also
checked for freshness. Equal model specs or independently current source files
alone do not authorize mixed checkpoints. No extra `load` call is made.

External CPU experts use the shared `ExpertMetadata` support validator, then a
separate retained-host ledger: each Native F32/BF16 linear retains both original
parameter storage and the deep-cloned Native storage, thus `2 * slice.bytes`.
Numeric FP8 CPU EP is rejected before preparation; it is not decoded or silently
routed to another provider. TP-local dense/shared matrices share their local
Arc allocation and use the model `rank_layout.weight_budget` ledger (including
norm copies), alongside state capacity and physical KV. Shared experts are not
charged to external EP owners.

`hybrid_tensor.rs` is the public factory/translation façade;
`hybrid_tensor/program.rs` owns model execution and child lifecycle. Its executor
binding constructor is private to this factory. The model call binding proves
caller consistency, not arbitrary executor authentication.

## Regression evidence

- `admission.rs`: F32/BF16 two-storage budgets, startup refusal between W and 2W,
  shared owner charges, numeric FP8 pre-materialization refusal, same-spec
  different-shard cleanup, and same-source parameter mapping mismatch.
- `schedule.rs`: TP1/2/4 translation of every exported site at both PP boundaries,
  S/R shape agreement, and sealed version/kind/semantic/shape/ordinal/layer equality.
- `../tensor_scope/per_sequence.rs`: real S=2 packed identity, exchange/preflight,
  and overflow checks.
- Existing `protocol.rs`, `startup.rs`, `child_lifecycle.rs`, `numerics.rs` and `custody.rs` exercise
  real PP2×TP2×EP2 state/logits/KV, rollback, cohort Ready, no partial publish,
  Unknown custody, nonzero source, sparse EP owners and partial startup cleanup.

The test entry module only wires `fixture.rs`, `probe.rs`, `numerics.rs`,
`custody.rs` and focused protocol/admission suites. Existing arithmetic and
rollback assertions were retained while moving them out of `mod.rs`.

`retry.rs` exercises public-factory TP1 compute cancellation and real EP errors,
complete journal rollback, then a new transaction with different tokens matching
the oracle; the same in-flight failures remain terminal at TP2. `custody.rs`
checks typed Unknown, one retained transaction, four credits, the complete four
KV-owner cohort, retained newly allocated page and unchanged committed frontier.
Repeated shutdown/release cannot wash out that evidence.

`tracing.rs` runs the real public factory and transport for success, failure and
Unknown, requiring the production allowlisted boundary fields and excluding
payload/error/path/session data. It isolates the tracing callsite registry in an
exact-test subprocess, not a substitute forward loop or hand-emitted events.
The previous tracing `E0382` ownership failure is covered by the same compile
path and the targeted tracing test.

`late_fault.rs` is the context/page-boundary matrix. Every case uses the real
factory and real `PipelineStageWorker` owners:

| Context before late fault | TP | Fault | Cases | Expected |
|---|---:|---|---:|---|
| 3 tokens, existing half page | 1 | owner error / owner cancellation | 2 | full rollback, no publish, TP1 retry succeeds |
| 3 tokens, existing half page | 2 | owner error / owner cancellation | 2 | full rollback, no publish, TP2 remains poisoned |
| 4 tokens, next decode allocates a new page | 1 | owner error / owner cancellation | 2 | new page returned, no publish, TP1 retry succeeds |
| 4 tokens, next decode allocates a new page | 2 | owner error / owner cancellation | 2 | new page returned, no publish, TP2 remains poisoned |

Total: **8 independent cases**, added to the existing 27 hybrid tests (35 total).
Each case checks every owner for state/KV
snapshot progression, active transaction count, expert outstanding count,
physical slot/free capacity and release state; checks parent committed tokens,
allocated pages, retiring pages, publication count, rollback rank coverage and
post-failure cleanup. The CPU pool snapshot is an observation-only read followed
by its own local abort; it does not call the parent protocol or publish.

For TP1 the next transaction uses a different token, compares logits against a
clean oracle and checks each owner's entry state/KV against its saved committed
prefix. For TP2 the retry is asserted to be
rejected without another owner execution, preserving the poison distinction.
The existing context-2 in-compute faults, typed-Unknown custody, rollback and numeric
coverage remain in their original suites.

Default-build compile-fail cases cover the config, stage, program and exact
factory signature; the reference test verifies the matching positive signatures.
