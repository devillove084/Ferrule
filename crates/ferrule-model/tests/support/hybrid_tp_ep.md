# Rank-local hybrid CPU-reference execution

```sh
cargo test -p ferrule-model --features cpu-reference --test hybrid_tp_ep
```

## Model contract

`HybridParallelPlan` is payload-free metadata. Its constructor accepts an ordered
TP roster and `Option<HybridExpertAttachment>`; the attachment contains an explicit
`source_rank`, ordered expert members and exact per-stage placement. A dense stage
has no attachment. EP members are a union with TP members, not another mesh axis.
Use `expert_attachment()` for presence, source and placement. There is no empty
placement sentinel and no placement-only compatibility accessor.

`plan.rank_layout(lane)` is the physical authority for local GQA/GDN geometry,
parameter partitions, compact state/KV schema and budgets. Global layer IDs remain
unchanged. Lane indices, global owner IDs and CUDA ordinals are not interchangeable.

`HybridTensorCpuSegment::prepare` uses the existing `StateDictMaterializer` and
`HybridDecoderSegment` layer loop. It reads catalog-bound local rectangles,
including discontiguous GDN Q/K/V channels, into decoded `PreparedCpuLocalMatrix`
storage. The parameter and linear share this allocation. There is no full prepared
projection retained beside a shard, no execution-time checkpoint decoding, no
weight slicing, and no lazy weight cache. GDN vector values are shared by `Arc`.
Embedding, router and scalar shared gate are intentionally replicated. Existing
norm storage retains source plus decoded values; `weight_budget()` charges both.
Routed expert payloads belong to the external expert owners, not this TP image.
The payload budget excludes RoPE tables, metadata, allocator overhead, activations,
collective buffers and workspace; runtime must charge those separately.

The CPU operators borrow prepared local values. Attention output and shared/dense
down partials reduce before residuals; the shared scalar gate is applied after
reduction. The vocabulary head uses the existing ragged column gather/unpack.
Only the explicit source rank dispatches routed experts. Other ranks contribute
zero and receive the weighted result through Sum. Each owner has its own KV pool,
even when KV heads are replicated. GDN key/value groups remain intact.

## Exact runtime integration API

1. Build `HybridParallelPlan::new(spec, range, scope, members, attachment)`.
   Validate it with `validate_spec`; obtain each `rank_layout(lane)`.
2. Compute `layout.weight_budget(resources, segment)`, `layout.state_bytes()` and
   `layout.kv_planes(page_size, max_positions)`. The weight read limit bounds the
   aggregate local parameter, not each subrectangle independently. It is not a
   total resident-memory admission limit.
3. Create endpoints with
   `DecoderTensorCollective::new_group_from_members(members, epoch, group, limits, timeout)`.
   Prepare the image inside each owner. `module.schema()` now has **local physical
   shapes**, with the same global layer and compact KV coordinates as the plan.
4. Use `HybridTensorSequenceState` and `module.lifecycle()` in the existing
   `PipelineStateProgram`/journal. Lifecycle wraps the existing `HybridLayerStates`
   representation and its create/frontier validation, adding owner/scope checks.
5. Runtime constructs its `ExpertParallelExecutor` from the exact attachment.
   On the source owner call
   `module.bind_expert_call(scope, attachment, &mut executor)` and pass the returned
   `HybridExpertCallBinding` to `module.execute(...)`; all other owners pass `None`.
   Binding rejects foreign scope/source/placement before execution. This checks
   caller-supplied attachment metadata, not the internals of an arbitrary trait
   object: authenticating the executor's actual worker roster remains runtime-owned.
6. `execute_in_transaction(transaction, ...)` additionally checks the caller's
   transaction against the entered KV view. Sequence IDs are in packed-sequence
   order. Delegate `validate_frontier`; keep install/publish/rollback and all
   Unknown custody in the existing journal.
7. TP1 bypasses peer collectives. After an error, a newer transaction is permitted
   only after the runtime journal proves rollback/quiescence. TP>1 errors poison
   the image/endpoint lifetime and wake peers; rollback does not resurrect it.
   Successfully computed but discarded work can use fresh checkout and a newer ID.

Runtime sealed integration exists in `PipelineParallelExecutor::new_hybrid_cpu_tensor`.
`translate_hybrid_schedule` translates the exported model wire sites and seals
version/kind/semantic/shape. The model's pre-write input/attachment exchange is
an additional content check, not a substitute for runtime's `TensorBatchIdentity`,
authenticated cohort or operator-site admission.

## Tests and boundaries

Model tests retain mathematical/layout/binding/prewrite/preparation/panic contracts
only. The old `support/hybrid_tp_ep.rs` and `support/hybrid_pp_ep.rs` Pipeline
command drivers, `RankProgram`, manual Ready/Install/Publish and synthetic dense
journal bindings have been deleted.

Shared code under `tests/hybrid_parallel/` is separated by responsibility:

- `model_contract.rs`: model preparation/call helpers and bounded host endpoints.
- `local_transaction.rs`: direct existing lifecycle and `PagedKvBackend` calls;
  no `PipelineStageWorker`, commit binding, participant roster or coordinator.
  `commit_local` only seeds the next numerical step; `rollback_local` disposes
  caller-owned test work. Neither proves a cross-owner decision.
- `experts.rs`: prepares per-owner providers, then borrows existing
  `CpuReferenceExpertWorker`, `CpuExpertResultExecutor` and
  `ExpertParallelRoutedExecutor`. No copied FFN/dispatch/combine math, runtime
  child workers or authentication claim. Materializers are dropped after prepare.
- `probe.rs`: read-only state/KV captures and `3e-5` comparisons.
- `rank_image.rs`: TP1/2/4 with actual owners `[7,19,31,63]`, nonzero source,
  no-EP dense, scope rejection and exact local rectangle accounting.

Fixtures under `tests/support/*_fixture.rs` supply deterministic checkpoint data;
they do not drive execution. The runtime fixture import path is unchanged.
Sparse IDs are model/collective contracts: model layout and endpoint owners agree,
but these tests do **not** claim sparse runtime KV-mesh admission. No owner ID is
rewritten to a dense lane for a journal.

PP/TP numerical tests compare logits, each local GDN conv/recurrent partition and
GQA KV history against the same checkpoint's unsplit `HybridDecoderSegment` for
prefill plus successive decode steps (including partial and new pages). Every
TP forward verifies unchanged checkpoint read counters. Cross-owner atomicity,
retry and Unknown custody are covered by the real runtime factory, not this
local mathematical setup; see [the migration map](hybrid_pp_ep.md).

CPU-reference currently admits TP1/2/4, identity unscaled F32, GQA/GDN and
bias-free/unclipped local projections. Local recurrent/KV state and rank-local
weight reads actually execute. The **model CPU-reference** method
`HybridTensorCpuSegment::validate_production_execution()` still rejects.
The runtime factory's separate
`Qwen35CompositionAdapter::validate_production_execution(profile)` (in
`ferrule_composition::model_factory::qwen35_composition`) checks the [restricted CUDA thread factory](../../../ferrule-runtime/tests/support/hybrid_cuda_tensor.md#typed-capability-and-production-validation).
No production GPU factory, CUDA operator, process transport, HostKV path or
production admission is enabled by these CPU-reference tests.

## Model-owned schedule and agreement V2

Runtime's existing `translate_hybrid_schedule` consumes this metadata instead of
reconstructing the layer loop or assuming a single-sequence agreement packet:

```rust,ignore
let schedule = parallel.collective_schedule(&segment)?;
assert_eq!(schedule.agreement_version(), HYBRID_AGREEMENT_VERSION); // 2
for site in schedule.wire_sites() {
    // site.ordinal: u64; site.global_layer: Option<usize>
    // site.semantic: HybridCollectiveSemantic
    // site.kind: HybridCollectiveKind::{AllGather, ReduceSum}
    let sent_f32 = site.shape.elements(sequence_count, packed_row_count)?;
    // Translate/register this site in the existing runtime sealed cohort.
}
```

The same schedule is retained by `HybridTensorCpuSegment::collective_schedule()`.
The execution Channel consumes it in order, checking semantic, global layer,
kind and actual payload length; it also requires all sites to be consumed.
`sites()` includes local logical sites for TP1. **`wire_sites()` is empty for
TP1**, including agreement and completion; runtime still owns sealed local
admission. No scalar shared-gate wire operation is emitted: scalar gating is
local math after the shared down reduction and before the routed output sum.

`HybridCollectiveShape { fixed, per_sequence, per_row }` means **F32 elements
sent by each rank**, not gathered result size:

`fixed + per_sequence * sequence_count + per_row * packed_row_count`.

All arithmetic is checked, including F32 bytes against `isize::MAX`. OutputHead
uses `ceil(vocab_size / TP)` per row, including ragged-lane padding. Agreement
and OutputHead use AllGather; all other sites use ReduceSum. Completion is one
fixed element. Attention/dense/shared/routed sites carry global layer IDs;
agreement/output/completion use `None`.

Stable tags and order:

| Semantic | Tag | Kind |
| --- | ---: | --- |
| Agreement | 0 | AllGather (0) |
| AttentionOutput | 1 | ReduceSum (1) |
| DenseMlpDown | 2 | ReduceSum (1) |
| SharedExpertDown | 3 | ReduceSum (1) |
| RoutedExpertOutput | 4 | ReduceSum (1) |
| OutputHead | 5 | AllGather (0) |
| Completion | 6 | ReduceSum (1) |

`encode()` returns payload-free registration words: header
`[version, degree, logical_site_count]`, then seven u64 words per logical site:
`[ordinal, global_layer_or_u64_MAX, semantic_tag, kind_tag, fixed, per_sequence, per_row]`.
It describes communication, not weight/model identity or authentication. Runtime
must still validate topology, actual expert owners and logical batch identity.

Agreement V2 serializes every integer as four little-endian u16 limbs in F32.
Its shared recipe drives both `encode_agreement()` and the Agreement shape:

1. Prefix: version, topology, replica, stage, layer start/end, TP count,
   ordered TP IDs, has-EP; if attached, source rank, EP count, ordered EP IDs,
   then per layer routed-count/top-k/expert owners; input tag (0 token/1 hidden),
   and next-layer only for hidden input.
2. Packed row count, sequence count.
3. Per sequence: session ID, context length, end length, phase (prefill=0/decode=1).
4. Per row, interleaved: token, position, sequence index.
5. Hidden F32 values, when present.

Therefore `fixed = 4 * (prefix_words + 2)`, `per_sequence = 16`, and
`per_row = 12 + hidden_width` (hidden width is zero for token input). Neither
physical page identity nor owner-local execution generation is serialized.
Local generation/custody checks remain mandatory before entering agreement.
Hidden next-layer/row count/width/F32 dtype are checked before packet allocation
or any wire call.

Compared with the old implicit encoding, V2 adds one version word and removes
one generation word **per sequence**. Thus total length changes by `4 - 4*S`;
for S=1 lengths happen to match, but packet ordering/content is incompatible.
Input tag/next-layer now live in the prefix, and row fields are interleaved.
The mixed four-layer TP2×EP2 fixture exports Agreement `(124,16,12)`, giving
176 elements for S=1/R=3 or 192 for S=2/R=3. The old S1-only expression is
`140 + 12*R`: identical at S=1, insufficient for S=2. Do not use passing S1 tests
as proof of equivalent encodings; runtime now seals the actual V2 coefficients.
A hidden dense stage exports
`(4*(12+TP),16,12+hidden_size)`.

`bind_expert_call`/`HybridExpertCallBinding` are the only call-binding API.
The old alias and entry point were removed. This checks caller consistency;
runtime constructs and validates its real `ExpertParallelExecutor`.

On TP>1, both errors and panics poison the module and abort the endpoint before
returning/resuming. Abort-hook panics cannot replace the original error/panic.
The original working state and entered KV custody remain with the caller; only
its existing journal may prove rollback/quiescence. A caught panic cannot make
that module reusable, even after caller-owned rollback and a newer transaction.
TP1 has no peer cohort to poison; any retry still requires caller-proven cleanup.

`hybrid_parallel/review.rs` tests the schedule, checked public ranges, two packed
sequences with different owner generations, hidden preflight and caught panic
with retained module/peer wakeup/newer-transaction rejection.
`hybrid_parallel/state_bounds.rs` validates conv and recurrent buffer address
limits without allocating the enormous buffers.
