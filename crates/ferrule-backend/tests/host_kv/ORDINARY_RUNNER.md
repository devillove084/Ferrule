# Opt-in ordinary-runner HostKV decode handoff

## Production selection, not a single-layer harness

CUDA production entry on
`RunnerMultiSessionExecutor<GenericDecoderRunner<HybridCudaDecoder>>`:

1. Reserve with the existing runtime `KvPageManager` and obtain
   `derive_append_prefix(&reservation, transaction, 0)`.
2. `prepare_host_kv_decode(&prefix, states, batch)` validates normal executor
   readiness/batch capability, binds the exact manager/backend pair while
   quiescent, and uses ordinary `prepare_multi_session_batch`.
3. `execute_host_kv_decode(&prefix, states, batch, &budget, block_tokens)` derives
   all layer capabilities from that SAME live reservation. The helper never
   executes layers itself. It invokes the existing prepared runner forward.
4. On successful forward, use ordinary backend `end_transaction(Publish)` and
   runtime logical commit/retirement. On failure, retain the reservation and drive
   backend abort first. Pending/error/Unknown is not permission for logical abort
   confirmation or resource-credit return.

`GenericDecoderRunner::execute_host_kv_decode` is also a production entry for
integrations that already construct a `HostKvAppendContext` from live borrowed
`AppendPrefixAuthority` implementations. Its constructor accepts those audited
capabilities, NOT `KvReservationView` or witness-only metadata. The existing
cross-crate unsafe authority contract has not been called sealed.

The normal path is:

`RunnerMultiSessionExecutor -> ReservedAppendPrefix -> all borrowed layer
capabilities -> HostKvAppendContext -> PackedTransactionRegistry -> same
DecoderTransaction::execute_scoped -> entered backend view ->
DecoderTransactionContext -> TransformerForwardExecutor::drive -> existing
CudaHybridModule::execute_layer -> CudaStandardKvBinding::with_context ->
execute_standard_layer_inner / execute_gqa -> real append DMA`.

The only loop remains `TransformerForwardExecutor::drive`. No second KV manager,
registry, page mapping, clone-based authorization, pointer cast or lifetime cast
was added. The default resident route remains unchanged.

## Scope and explicit policy

Supported: synchronous, one sequence, one decode token, nonempty committed
prefix, dense F32 **all-GQA** prepared model, all layers sequentially, full causal
attention. The CUDA hybrid composition is the existing model type, but this
opt-in policy rejects recurrent/mixed GDN layers, numeric profiles and EP/TP.

Prefill, empty-prefix, packed queries and unsupported models do not silently
fall back. A caller that wants resident prefill must explicitly use the existing
resident entry, publish it, and opt in for later nonempty-prefix decode.

CLI/server automatic scheduling selection is NOT enabled. No factory or GPU-owner
source was changed. To select this from the generic server driver rather than
from the production native executor API requires:

- an explicit per-owner policy/budget at composition, default resident;
- passing the live `PendingResidentKv::Reserved` token and authoritative manager
  borrow at `EngineDriver`'s existing prepared-execution call (currently it
  forwards only transaction/states/batch to `execute_prepared_batch`);
- a capability-aware optional `MultiSessionRunner`/native-executor dispatch
  surface, because the generic driver currently erases the concrete hybrid CUDA
  runner type;
- explicit routing of unsupported admission/prefill, with no fallback after a
  HostKV physical submission and no borrowed capability in a waiting payload.

A policy boolean or copied `KvReservationView` at the existing server call would
not supply authority. This change deliberately does not pretend otherwise.

## Same Entry, per-layer output lifetime

Each layer authenticates original source slots and staged shadow/COW/new-tail
write slots using the existing transaction identity/nonce/epoch. It downloads
actual post-RoPE Q, original prefix and staged tail, combines unnormalized
partials, and normalizes once, exactly as the previously connected append bridge.

After attention, the common standard layer performs the existing query gate and
attention output-linear. It drops the last scoped attention Rows, then calls the
unsafe `KvView::finish_gqa_consumers` boundary with the same operator owner. The
CUDA binding first drains owner submission roots using the existing barrier,
then records/synchronizes the backend final-consumer event. Resources and charge
are returned only after that proof. A `returned` flag prevents a never-submitted
or already-returned binding from being counted as a completed layer.

`HostKvAppendContext` advances the layer cursor only after the binding reports
retired custody; the backend's existing Entry read record is then idle. The next
layer obtains a fresh serial in that SAME Entry. Old serials cannot retire new
layer work. All layers must retire before normal output/forward completion.

Errors and panic drop live guards into the existing Unknown quarantine. A Waiting
result from the synchronous invocation becomes FailedActive, retaining the exact
continuation for normal cancellation. The borrowed context cannot be stored in
that continuation; the types provide no lifetime-extension path. The new CPU
continuation-gate tests check that conversion does not drop the held resources.

## Audited unsafe lint scopes

No blanket `-A`, crate-wide unsafe allow, removed Safety contract or weakened
completion check was introduced. Existing project code uses item-local audited
unsafe exemptions; this slice uses narrower lint expectations with explicit
`reason` text, so unused exemptions are also detectable:

- common `observe_complete` and `reject_submission`: physical proof integration;
- runtime `submit_trusted` and its rejection block: audited no-reference provider;
- runtime `unsafe impl AppendPrefixAuthority`: retained live manager/reservation;
- `KvView::finish_gqa_consumers`, its CUDA implementation, and the one internal
  standard-GQA call site: same-owner last-consumer retirement, including retained
  submission aliases;
- three private driver-free HostKV test modules: synchronous/no-work attestations;
- one ignored fixture fault adapter: borrows real runtime authority, delegates
  validation, and only injects an additional failure after physical admission.

Compile-fail tests reject safe calls to physical completion/rejection/trusted
submission and consumer retirement. Another proves the borrowed context cannot
be returned as a static continuation; the existing reservation-borrow test still
rejects logical abort while the capability remains in use.

## Validation status

No GPU was executed in this workspace session. The independent GPU owner did run
this fixture and recorded the result in
`target/validation/qwen35-35b-current/runtime-tp-ep-owner/SUMMARY.json`.

All long commands used at least 600000 ms limits. CUDA commands source
`target/validation/qwen35-35b-current/toolchain/env-cuda124.sh`, then override
`CUDA_VISIBLE_DEVICES=''`. No fake CUDA build or device query was used.

Passed:

- runtime `--test host_kv_append` without CUDA: 2 tests, including a real two-layer
  gated-GQA prepared CPU runner, prefill plus three decode steps, compared with
  whole-prefix execution using the same checkpoint tensors;
- model `--lib host_kv`: 2 synchronous custody tests;
- model `--lib transformer::forward::tests`: 5 existing wait/cancel regressions;
- backend `--test host_kv_protocol`: 23 tests, including next-layer admission only
  after consumer proof, old-handle replay rejection, and sticky late Unknown;
- common `--lib host_kv`: 13; runtime `--lib host_kv`: 30;
- CUDA-linked **driver-free** `host_kv_context_requires_all_layers...`: 1 test;
- model context / consumer-return and runtime reservation-borrow compile-fail
  tests: 1 each;
- CUDA runtime `check --lib --tests` and real `host_kv_append --no-run` in this
  workspace;
- independent GPU owner corrected-source run: `runner::cuda::ordinary_two_layer_append_matches_cpu_across_decode_abort_and_late_unknown`
  **passed 1/1 in 0.66s**, with clean GPU/custody status;
- independent GPU owner latest old append bridge run:
  `cuda::reserved_prefix_model_append_shadow_cow_rope_and_output_custody`
  **passed 1/1 in 0.47s**, with clean GPU/custody status;
- strict Clippy (`-D warnings`) for common lib/tests, backend lib/HostKV protocol,
  runtime lib/HostKV integration target, and CUDA model/runtime production libs
  and CUDA HostKV integration target;
- formatting and diff whitespace checks.

At the earlier runner-integration checkpoint, full four-crate
`clippy --lib --tests -- -D warnings` was actually run and was NOT green. It was
blocked by the following unrelated test-owner code at that time:

- factory test `crates/ferrule-composition/tests/unit/engine/model_factory/hybrid_cuda_pp_ep_factory.rs`
  (historical line 13): `option_map_unit_fn`;
- model `tests/unit/transformer/numeric_shard.rs:306`:
  `chunks_exact_to_as_chunks`;
- that model file at 394 and 420: `single_range_in_vec_init`.

These files were not modified or suppressed for this task. This historical
full-test result is not a claim about subsequent edits by those owners; the
fixture-budget follow-up reran only the targeted strict checks listed below.

## GPU owner handoff and exact fixture condition

The owner used the CUDA 12.4 environment and the isolated corrected source
snapshot under `runtime-tp-ep-owner/hostkv-source`. The first ordinary-runner
attempt failed before model execution because this pure-GQA fixture computed
`estimate.per_sequence_state_bytes == 0`, while production
`HybridCudaDevice::new_on_device` intentionally requires every budget domain to
be nonzero. This was a fixture-budget issue, not a production admission change.

The corrected fixture uses only:

```rust
state_bytes: (estimate.per_sequence_state_bytes * 16).max(1)
```

The production zero-domain rejection remains unchanged. The corrected owner
binary then passed:

```text
runner::cuda::ordinary_two_layer_append_matches_cpu_across_decode_abort_and_late_unknown
1 passed, 0 failed, 0.66s
```

The old append bridge also passed in the same owner SUMMARY:

```text
cuda::reserved_prefix_model_append_shadow_cow_rope_and_output_custody
1 passed, 0 failed, 0.47s
```

Provenance under `target/validation/qwen35-35b-current/runtime-tp-ep-owner/`:

- `SUMMARY.json` run `hostkv-runner-cuda-mechanical` and its `.stdout`/`.json`:
  exit 0, 1 passed, source_changed empty;
- `hostkv-mechanical-source-manifest.json` source SHA-256:
  `52b2670bd6b8e834f5af000de6b0bd5585d08fe84e38252de667b74e224421ca`;
- corrected runner binary SHA-256:
  `cba17b9803ac01d2b47ebf0d5dc913bb7504bfb4cd730a98aa1460a1822b06b0`;
- `hostkv-append-cuda-boundary` binary SHA-256:
  `005746f07576ef9b0df1217e900245f9cc287c5170df83c2fa260a173309fcc4`.

The complete-to-mechanical manifests differ only in `runner_fixture.rs`. The
owner run was against that pinned corrected source, not an arbitrary current
workspace. Comparing its fixture with the current workspace shows the same
budget expression, but the workspace additionally retains a prefix snapshot/COW
branch (capture before the first decode, release after publish). That branch is
compiled here but is **not** claimed covered by the corrected-runner GPU PASS.
The separate old append-bridge PASS must not be promoted into evidence for that
new ordinary-runner branch. No owner snapshot source was copied back except the
fixture-budget correction.

After synchronizing the correction, local validation passed with no GPU execution:

- `cargo test --locked -p ferrule-runtime --test host_kv_append`: 2 passed;
- `cargo clippy --locked -p ferrule-runtime --lib --test host_kv_append -- -D warnings`;
- CUDA 12.4 `cargo test --locked -p ferrule-runtime --features cuda --test host_kv_append --no-run`:
  real native test binary compiled and linked;
- CUDA 12.4 `cargo clippy --locked -p ferrule-runtime --features cuda --lib --test host_kv_append -- -D warnings`.

The synchronized workspace fixture SHA-256 is
`c81355bd8077e5739a1a93b3a57484c4772d27eff826601d0d0ee67789f76469`.
Targeted `rustfmt --check` for that fixture and `git diff --check` passed. The
follow-up `cargo fmt --all -- --check` reported concurrent, out-of-scope formatting
changes in `ferrule-cli/tests/unit/serve_resolution.rs` and factory tests now under
`crates/ferrule-composition/tests/unit/engine/model_factory/{hybrid_cuda_pp_ep_factory,hybrid_cuda_tp_ep_factory}.rs`;
none was reformatted here.

These targeted strict checks passed; they do not replace the earlier full-test
Clippy report above. Production admission, server policy, and all non-fixture
source were unchanged by this budget synchronization.

The ordinary fixture covers normal resident prefill, two-layer GQA decode, same
Entry per-layer custody, replay rejection, consumer budget return, cancellation,
and injected late authority failure. It does not test server policy or physical
VRAM release.

All long validation commands used at least 600000 ms limits. CUDA compile commands source
`target/validation/qwen35-35b-current/toolchain/env-cuda124.sh` and override
`CUDA_VISIBLE_DEVICES=''`. No GPU command was run locally.

## No VRAM-saving claim

Nothing here evicts a resident page or returns a fixed plane allocation grant.
Transient DMA/output custody retirement is not page placement. Existing snapshot
preemption can reuse a pool slot while keeping logical page identity, but physical
VRAM release still needs releasable page-level allocation ownership plus durable
all-layer host authority, budget, proofs, and active-write separation.
