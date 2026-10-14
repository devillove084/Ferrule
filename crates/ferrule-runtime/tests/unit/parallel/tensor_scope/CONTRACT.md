# Runtime TP endpoint and sealed-program contract

This is reusable runtime infrastructure, not hybrid production admission. It
adds no coordinator, transaction registry, model forward loop, or EP topology
policy. `PipelineStageWorker` remains the only physical/state custody journal.

## Endpoint construction

```rust,ignore
DecoderTensorCollective::new_group_from_members(
    members: &[ParallelRankId],
    epoch: ParallelTopologyId,
    group: ParallelGroupId,
    limits: HostCollectiveLimits,
    timeout: Duration,
) -> Result<Vec<DecoderTensorCollective>>
```

The public module is `ferrule_runtime::parallel::tensor::decoder_collective`.
Returned endpoint index and `owner()` follow the input member order, including
sparse/non-sorted IDs. The constructor rejects empty/duplicate/oversized rosters,
invalid timeout, and host allocation arithmetic overflow. Element/byte budgets
are checked by the host group for each exchange and for the entire sealed
schedule at `begin_sealed`. No model, weight, device or synthetic dense spec is
needed. The old `new_group(&StandardTensorPlan, ...)` delegates to this function.

The constructor only validates roster/storage constraints. Runtime admission must
still authenticate topology, PP stage, lane, model geometry and resources, and
assign epoch/group lifetimes. No process/GPU or hybrid factory gate is relaxed.

All endpoints now require an active `begin` or `begin_sealed` before exchange,
including the legacy unsealed path. Exchange before begin or after finish is a
terminal lifetime error. Do not silently create a new epoch or replay work.

## Schedule representation

Public types in `parallel::pipeline::tensor_scope`:

```rust,ignore
pub struct TensorCollectiveSite {
    pub site: u64,
    pub layer: Option<usize>,
    pub operator: TensorCollectiveOperator,
}

pub enum TensorCollectiveOperator {
    Linear(TensorParallelLinearPlan),
    Protocol {
        version: u64,
        operation: TensorCollectiveSemantic,
        kind: HostCollectiveKind,
        shape: TensorCollectiveShape,
    },
}

pub enum TensorCollectiveShape {
    Fixed(usize),
    PerRow(usize),
    Rows { fixed: usize, per_row: usize },
    PerSequence { fixed: usize, per_sequence: usize, per_row: usize },
}
```

`site.shape()` and `site.elements(sequences, rows)` expose the checked per-member
F32 input element count. `PerSequence` means `fixed + per_sequence*S + per_row*R`.
Both admission memory preflight and exchange take S/R from the complete sealed
`TensorBatchIdentity`; S comes from the validated packed sequence count, never
from rows. `element_count(rows)` has been removed; every caller must supply S and R explicitly. The parent currently admits S=1, while
the reusable collective/decorator accepts complete multi-sequence identities.

`Protocol` preserves model-exported kind and agreement version as well as the
semantic/shape; all fields participate in schedule registration equality.
Hybrid adapters translate `HybridParallelPlan::collective_schedule(...).wire_sites()`
only, and do not reconstruct the forward loop or agreement packet length.

Dense callers construct a real `TensorParallelLinearPlan` and use `.into()` (or
`TensorCollectiveOperator::Linear(plan)`). Full input/output shape, TP degree and
partition remain part of registration equality. Row partitions sum
`rows * out_features`; column partitions gather
`rows * ceil(out_features / TP)` elements from each lane. Column unpack/padding
remains the model's responsibility; dense vocabulary heads retain their real linear plan.

Semantic contract:

| Operation | Collective | Layer |
| --- | --- | --- |
| `InputAgreement` | AllGatherF32 | `None` |
| `AttentionOutput` | Model-exported kind | `Some(global_layer)` |
| `OutputHead` | AllGatherF32 | `None` |
| `GqaOutput` | AllReduceSumF32 | `Some(global_layer)` |
| `GatedDeltaOutput` | AllReduceSumF32 | `Some(global_layer)` |
| `DenseMlpOutput` | AllReduceSumF32 | `Some(global_layer)` |
| `SharedExpertOutput` | AllReduceSumF32 | `Some(global_layer)` |
| `RoutedExpertOutput` | AllReduceSumF32 | `Some(global_layer)` |
| `Frontier` | AllReduceSumF32 | `None`; fixed-only shape |

For routed output, the model adapter dispatches EP exactly once on the attached
leader; peers contribute zeros. InputAgreement validates encoded contents in the
model; runtime validates its site/kind/count and the complete packed identity.
Frontier is an optional model schedule marker, never a KV ACK or device fence.

Every owner registers the same immutable ordered schedule exactly once
before first admission. Empty schedules are accepted only for a single member,
whose model performs no wire exchanges; owner validation is still required
before finish. Runtime compares semantic operation, complete shape,
global layer, site, protocol version and kind, not only the resulting element count. A dense linear plan
must have the same degree as the endpoint roster. No owner can use legacy begin
once a schedule is registered. Missing registration, wrong topology/cohort,
wrong transaction, validation omission, reordered/wrong-kind/wrong-count calls,
and extra exchanges are rejected. Missing calls prevent finish and prevent a
scoped program from returning a successful execution result.

The model adapter must derive all sites from its actual operator order, including
input agreement, GQA/GDN projection reductions, dense/shared/routed outputs,
optional vocabulary gather and frontier marker. IDs are opaque (sequential IDs
are allowed); every exchange must use the matching ID and existing outer
transaction. Do not fabricate a dense plan to describe a semantic marker.

## State-aware program decorator

```rust,ignore
TensorScopedStateProgram::new(
    inner: P,
    control: DecoderTensorCollectiveControl,
    owner: ParallelRankId,
) -> TensorScopedStateProgram<P>
// Implements PipelineStateProgram where P: PipelineStateProgram.
```

It forwards the exact `SequenceState`, `Lifecycle`, `KvView`, plan, hybrid schema,
frontier validation, expert statistics/outstanding count and shutdown. There are
no additional Clone/Send bounds. Construct the program/backend on their owner,
then build the ordinary `PipelineStage` and `PipelineStageWorker`.

Parent flow is unchanged:

1. Register schedules during startup.
2. Prepare all owners using the existing journal, preserving physical custody.
3. Seal the expected `TensorBatchIdentity` from the existing `KvCommitBinding`
   and exact `LogicalExecutionIdentity`; use the shared cancellation flag.
4. Journal validation compares each owner's actual packed batch before backend
   enter, then forwards the model's own validation. Local physical indices and
   execution generations do not replace the global logical identity.
5. Execute rechecks the previously validated identity and requires that owner's
   full schedule to complete. Validation/execute errors abort and wake peers.
6. Parent collects all execution replies and calls control `finish(transaction)`.
7. Only the existing coordinator/journal drives Ready, Install/PollInstall,
   Publish, Abort/Rollback, Retire and Finish.

`TensorScopedProgram` remains the compatible dense `PipelineStageProgram`
decorator and has the same identity/schedule completion checks. Neither wrapper
registers a schedule automatically or makes an arbitrary program production-ready.
The existing transport failure observer remains necessary for panic/Unknown and
failures outside the wrapped execute call.

## Cancellation and termination

Stale abort cannot cancel a newer execution. Pre-admission cancellation consumes
no execution watermark. Active cancellation cannot report successful finish.
Multi-member abort, exchange error, timeout and dropped waiting peer wake waiters
and fail the collective lifetime permanently. Rollback/cleanup does not reset
that failure. A registered empty singleton schedule has no wire lifetime: a
known step error/cancel holds its admission until the parent's complete journal
rollback and logical retirement proof. Only that path may close the singleton
step, retaining its watermark. Unknown/Pending remain sticky; protocol failure
and dropped endpoints are never reset.
Host completion, control finish, empty waiter counts and endpoint drop are not
physical KV cleanup or GPU quiescence evidence. Unknown retains custody under the
existing owner/transport rules; do not replace it with an optimistic cleanup ACK.

## CPU-only validation

```sh
cargo test --locked -p ferrule-runtime --no-default-features --features cpu-reference --lib parallel::pipeline::tensor_scope
cargo test --locked -p ferrule-runtime --no-default-features --features cpu-reference --lib parallel::tensor::decoder_collective
cargo test --locked -p ferrule-runtime --no-default-features --features cpu-reference --lib parallel::pipeline::tensor_tests
cargo test --locked -p ferrule-runtime --no-default-features --features cpu-reference --test host_collectives --test pipeline_tensor --test hybrid_pipeline_state
cargo test --locked -p ferrule-composition --no-default-features --features cpu-reference --test hybrid_pipeline
cargo check --locked -p ferrule-runtime --no-default-features
```

Tests live in this directory: `collective.rs`, `semantic.rs`, `state_program.rs`,
plus the relocated existing `sealed.rs` and `pipeline.rs` regressions.
