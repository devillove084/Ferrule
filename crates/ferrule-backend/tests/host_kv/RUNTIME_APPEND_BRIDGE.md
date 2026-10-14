# Runtime/model append bridge: explicit hook and opt-in ordinary runner connected

The complete ordinary-runner entry, current policy, audited lint boundaries and
latest validation are documented in [ORDINARY_RUNNER.md](ORDINARY_RUNNER.md).
The call chain below describes the original lower-level scoped append hook,
which remains available alongside the ordinary-runner integration.

## Real connected call chain

1. `KvPageManager::derive_append_prefix(&reservation, transaction_id, layer)`
   returns `ReservedAppendPrefix`, borrowing the **existing** manager/reservation.
2. `prefix.bind_cuda_pool(&mut backend)` binds the manager/backend pair while
   physical custody is quiescent, before preparing this append. This is explicit
   trusted composition: legacy page IDs cannot prove which manager populated a
   pool. No active pool rebinding or extra logical read lease is permitted.
3. Prepare and enter the ordinary `PagedKvBackend` transaction with its real
   `PackedDecoderBatch`, then obtain the real model `CudaKvView`.
4. `prefix.with_cuda_append(AppendGqaBinding { view, batch, budget, block_tokens },
   operators, request, consume)` calls the unchanged model
   `StandardDecoderOperators::paged_gqa` boundary and the existing
   `KvView::append_and_attend_cuda` hook.
5. `CudaStandardKvBinding::with_append_prefix` constructs a separate, explicit
   append binding. The default resident binding remains Drop-free and unchanged.
   This matters: adding output Drop custody to that default type extended its
   mutable borrow into unrelated segment code. No lifetime cast was used to
   bypass the conflict.
6. The model's existing `with_f32_gqa_planes` callback hands out an opaque
   `CudaAppendTransaction` minted by the actual backend view. It carries the
   same pool identity, transaction ID, nonce and entered epoch. No decoder-private
   `inner` accessor, global registration table, or separate manager is needed.
7. After the plane callback releases its RefCell borrow, the ticket accepts the
   **borrowed logical authority**, not witness metadata alone, revalidates exact
   single-query batch/COW/schema/incarnation, admits resources and registers both
   submitted domains in the same `Entry`.
8. Narrow internal D2D writes append the actual post-RoPE K and V into the exact
   staged shadow/COW/new-tail row. Original committed prefix slots are never
   overwritten. The real producer event is recorded after these writes.
9. The same bounded transport downloads the actual post-RoPE Q once, the
   original committed-prefix K/V chunks, then that staged tail K/V row. Each
   successful payload must match the exact immutable transport ticket/event.
10. The CPU consumer uses that one downloaded Q for both operands, accumulates
    prefix and tail **unnormalized** partials at the appended query position,
    normalizes once after the tail, and uploads the result using exact H2D proof.
11. Output readiness transfers a guard, NOT an ACK. Output allocation, operands,
    transport, six-domain charge and unresolved consumer record remain in `Entry`.
    `leave`, commit, abort completion and whole-plane mutable access stay blocked.
12. The runtime runs the scoped downstream consumer, consumes the exact returned
    model `Rows`, and records/synchronizes the actual last-consumer owner event.
    Only then does the original read record finish, resources drop and the charge
    return. A consumer error, panic, dropped guard, canceled operation or failed
    driver/event proof conservatively quarantines custody. Later success cannot
    rehabilitate Unknown. There is no cancellation recovery reaper in this slice.

## Authority and borrow boundary

`AppendPrefixAuthority` is an unsafe-to-implement cross-crate trust interface,
not a safe witness constructor. Its sole production implementation is runtime's
borrowed `ReservedAppendPrefix`, with a documented safety justification. There
are no raw pointer casts, transmute, unsafe lifetime extensions, public physical
transaction constructors or independent ownership ledgers in the bridge. The
unsafe integration remains explicit. Item-local `expect(unsafe_code, reason = ...)`
now records the audit rationale; no global lint suppression was added.
The trait is deliberately not sealed across crates: another adapter can provide
an audited `unsafe impl`. Its `validate` must revalidate live reservation authority,
not just witness geometry or cached success. Witness, source, transaction,
reservation and layer must remain stable throughout polling and output release.

`ReservedAppendPrefix::for_layer(layer)` derives a new borrowed sub-capability
from that same live reservation without mutating the installed authority or
registering another reader. The opt-in ordinary runner now constructs all such
layer capabilities for one decode, passes them through the existing typed
transaction context, and retires each layer before entering the next one. It does
not authorize concurrent operations.

The internal read record can move from an authenticated view to an owned opaque
handle without borrowing a temporary transaction object. It retains the SAME
physical identity/nonce/epoch and Entry pins; this is not a new KV transaction.
The public operation/output guard still borrows the runtime authority.

Caller Q/K/V allocations remain caller-accounted, as in the committed path.
Their retained aliases keep memory alive, not permission to overwrite input
while a DMA is pending/Unknown. The model hook is synchronous; it does not return
before those input transfers finish. The lower-level backend async interface
requires immutable Q/K/V until readiness, or retaining them on Unknown.

The downstream callback must enqueue consumers on the same operator owner and
must not retain output aliases. If it allocates another output, that allocation
requires its own charge. Automatic alias-aware transfer of grants into arbitrary
`Rows`/continuations is NOT implemented. This is a scoped output-life contract,
not permission to launch an unrelated stream and then ACK completion.

## Completion metadata is not physical evidence

`HostKvCompletion` remains public, copyable routing/diagnostic metadata. Neither
matching numbers nor a provider error can prove that physical references ended.
Common cannot depend on backend, so the audited adapter boundary is explicit:

- `HostKvPlacementOwner::observe_complete` is unsafe and requires exact backend
  stage/source/operation proof, including downstream consumer completion.
- `reject_submission` is unsafe and requires positive no-retained-reference
  evidence for the exact initial attempt; cancellation or a generic error is
  insufficient. Rejection retains the generation high-water mark, so retry must
  use a strictly newer generation even if stream/event numbers are recycled.
- Safe `ReservedHostKvPlacement::submit` treats every provider error as sticky
  Unknown and retains lease/budget. Acceptance is not completion. Provider panic
  after registration also marks Unknown before resuming the unwind.
- Only unsafe `submit_trusted` can interpret an audited non-Unknown provider error
  as no-reference rejection. Unknown, including later success after Unknown,
  never opens cancellation/refund. Pre-provider validation failure submits no work.

There is no safe unchecked receipt constructor, second manager or new registry.
The live backend bridge continues using its opaque transaction/DMA/output guards;
it does not convert common metadata into physical ACKs. Driver-free tests use
explicit unsafe attestations only for modeled no-work stages. Compile-fail tests
reject safe completion, rejection and trusted-submission calls.

## Supported and deliberately unsupported

Connected now:

- Explicit runtime-to-model invocation of the real append/attend operator hook.
- Opt-in `RunnerMultiSessionExecutor` ordinary-runner entry using the existing
  prepared transaction and layer cursor; default execution remains unchanged.
- One F32 GQA query, one sequence, nonempty committed prefix, all selected GQA
  layers in sequence.
- Partial-tail shadow, COW and page-aligned new-tail placement.
- Full causal attention at the append frontier, same actual post-RoPE Q,
  bounded original-prefix and staged-tail DMA, CPU partial merge, device output.
- Shared six-domain budget and original transaction fences/epoch/pins.
- Scoped downstream consumer and sticky fail-stop cancellation/Unknown.

Policy boundaries:

- No default CLI/server policy switch was changed. The production selector is the
  explicit `RunnerMultiSessionExecutor::prepare_host_kv_decode` followed by
  `execute_host_kv_decode` entry; callers must opt in after obtaining the live
  `ReservedAppendPrefix`. Existing engine/server calls retain the resident path.
- The typed authority is borrowed only in the synchronous forward call. If the
  runner would return Waiting, the append route converts it to FailedActive while
  retaining the exact continuation for normal abort; it never stores the borrow
  in a continuation. Unknown/error/drop leaves Entry custody and budget charged.
- Existing segment construction still calls the default resident binding unless
  an owner explicitly chooses the new ordinary-runner entry. A normal `Rows`
  return alone cannot refund this grant safely.
- Packed/multi-query batches, empty-prefix first prefill, BF16/hybrid/numeric paths,
  parallel stage topology routing, GPU-computed tail partial, async model waiting.
- The model GQA request has no sliding-window field, so this bridge supports full
  causal GQA only. CPU production-stream tests additionally exercise windowed
  prefix/tail math; that does NOT enable windowed model dispatch.
- No Unknown recovery, page eviction, logical frontier publication or VRAM saving.

The ordinary-runner follow-up adds narrow hooks in decoder runner/runtime/
transaction, the existing transformer loop, standard GQA composition, operator
consumer-return boundary and the full-model CUDA hybrid adapter. It does not
change `standard/cuda.rs`, decoder `kv/cuda.rs`, segment, hybrid_rank, numeric
binding, binding, materialize, TP or factory owners. Existing unrelated dirty
work was not reverted. No commit.

## Pool slot reuse is not physical VRAM-grant release

This bridge neither detaches a resident mapping nor preserves a durable all-layer
host page image. It retains transient chunks/partials. Successful copy/attention
must not lower the resident-page count; the GPU fixture checks unchanged capacity.

The existing quiescent whole-page `preempt` can preserve logical page identity in
its snapshot store and return a slot to the SAME pool. That supports slot reuse,
not physical VRAM release: fixed-capacity per-plane device allocations stay live.
The next page-placement step needs complete all-layer host authority and budget,
all required DMA/consumer proofs, active-write/original separation and host-aware
read/restore routing within the same transaction/snapshot store. Returning actual
VRAM also requires releasable allocation granularity/physical grant ownership.

## Validation in this change: no GPU execution

CPU (600000 ms command limits):

- Backend `--test host_kv_protocol`: **23 passed**. Includes exact production
  appended-tail stream math, same downloaded Q, windows, ticket identity,
  malformed payloads, Unknown, original/shadow/COW/epoch and budget contracts.
- Common `--lib host_kv`: **13 passed**, covering ordered completion, Unknown,
  exact fields/owner, rejection retry generations, replay and budget custody.
- Runtime `--lib host_kv -- --test-threads=1`: **30 passed**, including layer
  sub-capability, append-contract and provider error/panic/trusted-rejection cases.
- Common `--doc host_kv`: **2 compile-fail tests passed**.
- Runtime `--doc submit_trusted`: **1 compile-fail test passed**.
- Runtime `--test host_kv_append` without CUDA: **2 passed**, including the
  two-layer ordinary-runner CPU reference (prefill plus repeated decode).
- Runtime model continuation custody tests: **2 passed**; backend per-Entry
  cross-layer retirement/late-Unknown tests: **23 protocol tests passed**.
- Runtime `--doc ReservedAppendPrefix`: **1 compile-fail test passed**, proving
  abort cannot consume/mutate the live borrowed reservation.
- Backend CUDA `--doc AppendPrefixAuthority`: **1 compile-fail test passed**,
  rejecting a safe metadata-only authority implementation. No GPU program runs.
- Initial runtime lib-test link failed with undefined hidden LLVM symbols in
  unrelated standard/Tokio/process code. `CARGO_INCREMENTAL=0` rerun passed in
  4m45s. No cache deletion or unrelated source edits were needed.

Formal CUDA environment (no fake/stub build):

```sh
. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
CUDA_VISIBLE_DEVICES='' CARGO_BUILD_JOBS=2 cargo check --locked -p ferrule-runtime --features cuda --lib --tests
CUDA_VISIBLE_DEVICES='' CARGO_BUILD_JOBS=2 cargo test --locked -p ferrule-runtime --features cuda --test host_kv_append --no-run
```

Both passed; the latter compiled and linked the real native-CUDA fixture,
including the ignored two-layer ordinary-runner HostKV fixture and its explicit
opt-in executor entry (not GPU-executed in this workspace validation).
The final security-review rerun also passed backend CUDA `check --lib` and runtime
CUDA `check --lib --tests`. An earlier runtime rerun was blocked by unrelated
factory `Clone` bounds in the then-named `hybrid_cuda_tensor.rs` (now
`crates/ferrule-composition/src/model_factory/hybrid_cuda_tp_ep_factory.rs`);
that blocker was absent on the latest workspace snapshot, without HostKV edits
to that file. The source script
selects GPU 0 by default, so every CUDA command overrides visibility AFTER
sourcing. No GPU tests, nvprof, benchmark or device queries were executed locally.

The independent GPU owner has since passed the old append-bridge regression
(1 test, 0.47s) and the two-layer ordinary-runner fixture on its corrected source
snapshot (1 test, 0.66s). The correction was the fixture's nonzero state budget:
`(estimate.per_sequence_state_bytes * 16).max(1)` for pure GQA; production admission
was not changed. Both tests remain ignored by default and were not GPU-executed
locally. See [ORDINARY_RUNNER.md](ORDINARY_RUNNER.md) for exact SUMMARY entries,
source/binary hashes and the source condition: the current fixture's additional
prefix snapshot/COW branch is compiled, but not covered by that runner GPU PASS.

Focused strict Clippy passed for common/backend HostKV targets and model/runtime
production libraries, including the CUDA HostKV test target. At that validation
checkpoint, full workspace strict test Clippy was blocked only by pre-existing
unrelated test-owner lints in factory tests (now
`crates/ferrule-composition/tests/unit/engine/model_factory/hybrid_cuda_pp_ep_factory.rs`)
and model `tests/unit/transformer/numeric_shard.rs`; those files were not changed.
