# Owner-local root submission segments

## Implemented and enabled

`standard/cuda/root_submission.rs` is the CUDA-independent custody ledger.
`root_execution.rs` is the concrete CUDA implementation. No transaction is
created. No global fence-disable flag is introduced. `consumer_proof` remains
an unconditional real consumer proof when called; only the explicit
`ordered_consumer_proof` seam may defer same-stream workspace-reuse proofs.

Already-resident, bias-free, non-aliased non-expert projections in owners with
consumer-proof accounting enter a finite
one-projection scope through the existing `StandardDecoderOperators::linear`
implementation. Cold/mixed/unsupported paths retain the synchronous path. The
existing numeric expert and dense SwiGLU scopes are preserved. Plain F32
KeepAll standalone projections retain their old path (there are no redundant
model consumer events there); they can still participate in a layer scope.

Within a root layer scope, RMSNorm, projection, query/gate split, sigmoid gate,
RoPE, residual, and eligible resident dense/shared SwiGLU can share completion
proofs. These use the existing arithmetic kernels. Eligibility does not depend
on stored weight encoding or numeric precision. Current activation roots remain
F32, as required by the standard Rows contract; this does not add a W8A16 backend
or change its arithmetic.

## Production layer integration

The CUDA owner has this concrete method:

```rust
pub(in crate::transformer) fn execute_root_layer(
    &mut self,
    materializer: &StateDictMaterializer,
    layer_index: usize,
    attention: standard::hybrid::AttentionBlockRef<'_>,
    feed_forward: &PreparedFeedForwardBlock,
    kv_layer: usize,
    hidden: &mut CpuTransformerHidden,
    kv: &mut dyn KvView,
    states: &mut [&mut dyn StandardSequenceState],
    routed_execution: Option<&mut standard::RoutedLayerExecution<'_>>,
) -> Result<()>;
```

This calls **the one `execute_standard_layer_inner`**, with exactly the same
arguments. It does not duplicate layer composition. `cuda/hybrid.rs::execute_layer`
enters this seam directly. CUDA standard and hybrid segment owners select the
private `LayerOperators::Cuda` adapter; their shared segment loops still perform
all transaction, identity, precision, state and frontier checks in the original
order. Public trait-object/CPU segment entry points select `LayerOperators::Standard`
and retain ordinary synchronous layer composition. No public asynchronous Rows
or root-scope API is introduced.

The lower-level closure API `with_root_submission` is private to CUDA ownership.
Its byte allowance must be a subset of already admitted operation/activation
scratch, never an independent budget. The concrete layer method computes the cap
itself from the maximum existing projection/norm/SwiGLU operation envelope, not
the sum of layer intermediates.

### External EP producer boundary

The shared FFN composition calls the default no-op operator hook
`before_external_routed_execution` immediately before invoking the external
`routed_swiglu`. CUDA maps it to `root_barrier(ExternalExperts)`. A failed proof
prevents both executor dispatch and the liveness callbacks it can invoke. The
local expert-provider path is unchanged. The former whole-layer external-EP
fallback is removed only because this boundary is now wired.

Entry/inter-layer liveness callbacks, external expert drain, KV leave and
transaction working-copy cloning remain outside the lexical layer scope. The
existing external executor holds/drain contract remains authoritative for remote
consumers; a local producer proof is not remote completion evidence. Errors and
panics still pass through root quarantine and existing owner handling.

For a future recurrent state accessor that clones/rebinds storage, add the
analogous default `before_recurrent_state_access(&mut self) -> Result<()>`
before `states.iter_mut().map(|s| s.linear_state(layer))`; CUDA maps it to
`RecurrentState`. Current CUDA `linear_state` only borrows the state. Actual
`gated_delta_net`, scratch growth and state position publication remain fully
synchronous. Transaction working-copy cloning and KV leave occur outside the
lexical layer scope and must remain outside it.

## Hard boundaries and custody

Explicit reasons cover D2H, CPU router, external EP, TP, KV publish, recurrent
state, workspace growth, cache eviction/rebind, diagnostic observers, pin
capacity, scope exit and unclassified operators. `synchronous` always flushes
an outstanding root segment before running an unclassified operation. KV's
existing independently owned proof/leave semantics are unchanged. TP and
installed observers conservatively disable the multi-operator scope.

Growth inside an operation proves completion but cannot release roots: later
kernels in that same operation may still consume them. Segment retirement is
only allowed between semantic operations, after real consumer/transfer proofs.
Allocation or upload events are never accepted as consumer completion.

Any in-scope local operation error, panic or Unknown poisons reuse and retains
roots, resident image artifacts, workspace, cache credit and owner authority. A
later successful stream sync cannot clear this state. Scope drop leaks
quarantined roots consistently with existing Unknown custody, rather than
freeing them.

An ordinary external callback error (including Waiting translated to an error)
is different: after a successful `ExternalExperts` producer boundary, with no
pending local roots and no owner quarantine, the lexical scope returns the
original error and releases only its local scratch reservation. The external
owner retains its independent holds/KV authority and continues its existing
Pending/Complete/Unknown drain protocol. This is not remote completion evidence.
The ledger explicitly records the external handoff even if a synchronous shared
gate already proved the producer, without overwriting the last actual proof's
`last_barrier`. Local operation entry or any other boundary revokes the handoff;
proof loss and panic remain sticky Unknown. No error-time stream sync can turn
an unproven submission into this clean external-error case.
Capture-safe mode is rejected before the closure. An entry upload-stream sync
also checks the backend's actual capture state for lower-level graph callers;
there is no newly supported allocating capture path.

## Budget and pin maximum

- Fixed array: **16 activation root views**, no growable root pin vector.
- **16 semantic operations** maximum between barriers, including operations
  whose exact views deduplicate.
- Checked byte cap `B`: `sum(retained view bytes) <= B`. Forecast includes every
  input/output; conservative duplicate forecasts may flush early.
- Numeric cache reservation remains `workspace_reserved + max(B, operation)`
  throughout the lexical scope, including between host calls. The original
  reservation is released after successful scope exit or a clean external-error
  exit with already-proven local completion. Unknown freezes it.
- Root image weights remain in their existing immutable binding inventory.
  The root path does not create another expert cache or stage whole layers.
- Existing model-owned full activation buffers are used. Arbitrary external
  suballocation views with unrelated backing-storage ownership are not a new
  supported layer-scope input. Allocator grant/segment slack remains the existing
  inventory/admission responsibility, not an allocation-event completion proof.
- The concrete layer cap is a max of ordinary per-operation payload plans and
  fits within the existing conservative `32 * width * rows * 4` activation
  envelope. Recurrent scratch and numeric workspace stay separately reserved.
  This is a logical bound, not a measured allocator peak or new GPU admission
  validation result.

## Tests and measurement status

CPU tests compile the production ledger through `tests/root_submission.rs`:
finite byte/root/operation limits, exact-view deduplication, barrier retirement,
sticky Unknown and absence of allocation/upload completion transitions. A small
source guard additionally protects the opaque-operator fence and the unique
layer implementation call.

The CUDA unit tests in `root_execution_cuda.rs` are GPU-required and ignored by
default. They compile as part of the model CUDA library tests. They assert:

- Warm norm + two projections + residual + RoPE: model consumer-proof attempts
  **9 -> 1**, with identical F32 output and four roots/144 payload bytes under
  the fixture's already admitted 208-byte scratch envelope.
- D2H and CPU router: producer flush plus the ordinary host-call proof remain.
- Full pin cap and workspace growth produce real additional barriers.
- Error, panic, allocation failure and lost proof retain custody after a later
  successful sync, including after owner drop.
- Dense SwiGLU uses five bounded roots; a byte-full FFN flushes before residual.
- Unsupported capture never invokes the closure.

These are test assertions, **not results from GPU execution**. No new v4
EventSynchronize/token, latency, throughput or physical peak result is claimed.
The production layer callers and external EP hook are now integrated. The
CPU-runnable `root_layer_integration.rs` wiring guards protect CUDA owner opt-in,
unchanged default dispatch, and callback boundaries. Private FFN tests in
`external_routed_execution.rs` exercise dispatch/liveness ordering, barrier
failure, cancellation and the unchanged local route. Ignored GPU tests in
`root_execution_cuda.rs` check the exact trait hook's real consumer proof,
sticky proof-loss custody, clean external Failure/Waiting credit release after
both pending and already-idle producer boundaries, and rejection of stale
handoffs after local work, panic or owner quarantine. The existing
`hybrid_cuda/external.rs::external_failure_and_waiting_retain_module_cancel_and_root_pins`
continues to require abort Pending, retained pins, then Complete after explicit
remote drain; it also asserts no local quarantine along this recoverable path.
GPU execution and performance evaluation
against the 35,842/32-token run remain required; integration is not a measured
latency, memory or throughput result.
