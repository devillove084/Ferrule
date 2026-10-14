# Prepared BF16 attention — model-ready backend contract

Status: the three model-integration blockers are resolved and bounded GPU gates
pass on CUDA12.4/sm86/cuBLAS120405. No model files, reference values, tolerances,
precision profiles or default dispatch changed.

## 1. Warm per-sequence shapes, compose arbitrary bounded batches

```rust
use ferrule_backend::cuda_plan::bf16_attention::{
    Bf16AttentionPlan, Bf16AttentionShape,
};
let shapes = (1..=8)
    .flat_map(|m| (m..=128).map(move |n| Bf16AttentionShape { rows: m, tokens: n }))
    .collect::<Vec<_>>();
let plan = Bf16AttentionPlan::from_shapes(
    &shapes, 8, 2,       // maximum TOTAL query rows and sequences
    16, 2, 256,         // query heads, KV heads, Dk=Dv
    36 << 20, 16 << 20, // physical storage budget, external reserve
)?;
let mut prepared = op.prepare_bf16_attention(plan)?;
```

This is **996 distinct (M,KV) shapes**, not a Cartesian product of multi-sequence
cohorts and not 996 allocations. Fixed H/Hkv/D belongs to the plan. Only declared
shapes are warmed, deduplicated and sorted; both matrix calls are exercised twice
per shape, before invocation. One allocation and one bound handle/workspace are
reused. No autotuning or dummy maximum-matrix padding is used.

`plan.view(&groups)` accepts any composition of warmed shapes whose ordered
`{row_start, rows, sequence, tokens}` groups partition query rows within
`max_total_rows()` / `max_sequences()`. A sequence is always one complete head-
batched GEMM. New sequence combinations do not need prior tuple enumeration.
Malformed/unwarmed/out-of-bounds views reject before native submission.

`new(groups,...)` remains a convenience constructor. `with_views(cohorts,...)`
extracts per-sequence shapes and maximum packed capacities for compatibility;
it no longer requires exact whole-cohort membership. `max_rows()` means maximum
per-sequence M for scratch sizing; `rows()` describes the default view only.
KV lengths must be in 1..=1024 and each M<=KV. No fallback beyond this bound.

## 2. Use the private-plane custody adapter

Keep the existing `CudaKvView::with_bf16_gqa_planes` authority boundary. Model
code does NOT acquire mutable cache planes, slot arrays or raw native pointers.

```text
status = op.zero_i32_buffer(1)            // charge/retain as caller resource
prepared.begin_invocation(op, &mut status)
for each GQA layer:
  kv_view.with_bf16_gqa_planes(op, layer, authenticated_batch, |mut planes| {
    layout = planes.attention_layout(H, Hkv, D, scale)
    planes.append_and_attend_prepared(rows, layout, &mut prepared, &mut status)
  })
prepared.finish_invocation(op, &status)
only if all calls succeed: publish output / commit transaction working state
```

New adapter methods on `CudaBf16GqaPlanes`:

- `prepared_attention_view(&Bf16AttentionPlan) -> Result<Bf16AttentionView>`:
  derives the view from this guard's authenticated batch, no device work.
- `append_and_attend_prepared(CudaBf16GqaRows, layout,
  &mut CudaPreparedBf16Attention, &mut CudaI32Buffer) -> Result<()>`:
  validates plane/layout/owner/aliases, derives the composed view, validates
  transaction row metadata, uploads private metadata, then submits attention.

The adapter requires rows packed in sequence order with each query being the
contiguous suffix of that sequence's active KV. It retains existing COW/private
working-page authority, physical-writer checks and stale-view protection. It
neither clears nor consumes status and does not perform a status download.
Metadata is allocated/uploaded inside this adapter just as in the existing
append/attend pattern; arithmetic `execute[_view]` remains allocation/sync-free.
Metadata follows the allocator's stream retirement/Unknown quarantine rules.
Budget metadata/status separately, including simultaneous pending layer metadata
and existing allocator segment slack. The attention scratch grant does not cover
these buffers. With page_tokens16, capacity8/maxseq2/context128, each layer's five
metadata buffers is at most five 256-byte small grants; status is another 256-byte
grant. Do not infer that retired-but-pending buffers are immediately reusable.

## 3. Normal model allocations are allowed during invocation

There is **no invocation free-memory query or free-memory-decrease quarantine**.
Activation, metadata and expert allocations are legal between begin and finish.
The previous false-failure policy and its test were removed. A replacement GPU
gate allocates a live extra 4 MiB after begin, finishes successfully and reuses the
same prepared storage/handle.

Global free-memory changes during preparation are diagnostic only, not proof of
attributable provider allocation or a vendor hard cap. Attention no longer rejects
warmup based on those deltas. Persistent physical/external ledger charges and
up-front warming of declared shapes are the admission mechanism. The 16 MiB
external reserve is a caller charge/headroom allowance, **not a proven hard
cuBLAS allocation cap**. No post-warm runtime-allocation cap is claimed.

## Sticky status and retained downloads remain mandatory

`begin_invocation` explicitly clears one status word and retains its allocation
lease. A second begin rejects until `finish_invocation` consumes the previous
status. Execute never resets the latch. Invalid metadata, page addresses or
non-BF16-expanded Q are checked before append. Later layers cannot erase an early
error or write cache/output through a latched error. Vendor GEMMs may still touch
private scratch; semantic no-op does not mean no kernels submitted.

Finish uses the shared owned D2H completion boundary. Copy/fence failure marks
terminal Unknown and retains the host destination, device source and provider
custody. The next submission is blocked. A nonzero semantic status rejects the
invocation without pretending CUDA completion was lost. Caller aborts working
state on any adapter/execute/finish error. No implicit rollback is provided.

The shared runtime helper also serves GemmEx downloads. Borrowed host destinations
are updated only after successful owned staging completion. Preflight failures do
not poison. Unknown is never repaired by a later fence. Capture rejects before
submission. Existing `is_poisoned()` / `preparation_completion_unknown(op)` apply.

## Budget and arithmetic

Shared admission is keyed by exact stream owner and provider family, coordinated
by a short per-device initialization lock. Attention and projection can coexist:
16+16=32 MiB external charge plus their separate physical grants. Charges persist
through live handles, retained graphs and Unknown; clean retirement releases them.
Startup requires `CUBLAS_WORKSPACE_CONFIG=:0:0` before provider initialization.

For the example H16/Hkv2/D256, maxM8/KV128:

- Requested attention storage: **35,848,192 bytes**.
- Dedicated physical grant: **37,748,736 bytes (36 MiB)**.
- External reserve: **16,777,216 bytes (16 MiB)**.
- Combined attention charge: **54,525,952 bytes (52 MiB)**, before metadata/status.

One set of aligned Q/K/V/score/probability/PV panels plus a 32 MiB provider workspace
is reused across sequences and layers. Actual calls preserve each sequence's M/N
and full head batch. BF16 A/B/C, F32 accumulation, no TF32; score/mask/probability
boundaries and persistent-softmax tree unchanged. Output is BF16-expanded F32.
Finite-input/overflow or universal HF dispatch parity beyond tested cases is not
promised. The full model's 7 GiB fit remains its integrator's budget responsibility.

## Validation and evidence

`target/validation/bf16-attention-integration-ready-20261007/`:

- 13 focused GPU gates pass, including actual transaction/COW adapter, legitimate
  allocations, one-/two-device admission coexistence, sticky status and retained
  status/output copy/fence Unknown.
- H16/D256 capacity8/context128/maxseq2: 996 shapes, one owner, actual and KV128
  independent HF outputs exact. Separate same-owner M1/KV11, M1/KV128 and M5/KV129
  regression also passes twice.
- 24 original native cases: **312/312 stage/repeat comparisons exact**; both typed
  fixture/resource gates pass. Actual raw/gated4096 remain exact (original gated
  mismatch196 -> 0). No oracle/tolerance changes.
- 258 changed-view executes retain stable allocator metrics and no execute sync.
- Five CPU plan tests and the custody composition test pass; strict backend
  `--tests` Clippy passes with `-D warnings`.

Authenticated supervisor receipts include drain/ECHILD and three final all8-idle
samples. No full-model execution or latency/throughput claim.
