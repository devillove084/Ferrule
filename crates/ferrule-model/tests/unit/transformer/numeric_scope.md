# Owner-local numeric SwiGLU scope audit

## Existing duplicated proof and allocation lifetimes

Before this change, `numeric_linear_into` called `consumer_proof` before
workspace reuse/replacement and after each numeric projection. Thus gate/up/down
each had two model-level proofs. `admitted_expert_operation` separately proved
completion before clearing activation holds and completing its cache lease.
`routed_swiglu_batched` then submitted scatter and proved again before truncating
route holds. Finally `synchronous` proved the final operation and preserved its
boundary drain, compute-stream sync and upload-stream sync.

Those are distinct call sites, not measured CUDA EventSynchronize totals. In
particular D2H transfers and allocator retirement have their own evidence and
must not be counted as eliminated model proofs.

- `resident_linear_buffer` pins input/output through `expert_hold`.
- The numeric workspace belongs to `NumericState`; standalone replacement first
  proves completion, drops old storage and allocates the replacement within the
  existing reservation. An allocation error after that fence is a known failure.
- The existing cache lease covers all three canonical projection identities.
  Only completed/idle entries are evictable. The new `expert_artifacts` roots pin
  all three `Rc<ResidentLinear>` handles before any scope allocation/submission.
- The scope preflights all three existing `NumericFp8LinearPlan`s and reserves the
  maximum workspace. All four outputs (gate/up/product/down) and input are held
  before the first projection. No buffer replacement occurs between projections.
- The existing semantic numeric op, activation and scatter submit to the same
  `CudaOperators` compute stream without a host await in those primitives. The
  workspace is overwritten only after its prior consumer in that stream.
- Scatter now runs inside the cache admission closure; EP worker D2H likewise
  runs before completing the expert lease. Cross-stream D2H producer events and
  host completion waits are unchanged.
- Growth still takes a real `consumer_proof` before old workspace destruction.
  Unknown fails before replacement. On unknown or panic, owner roots, workspace,
  route operands, cache and charged credit follow the existing quarantine path.
  A later successful sync cannot clear the poison or refund credit.

## Scope and compatibility

The optimization is private and lexical: no flag that globally suppresses
proofs, no new transaction/state machine, and no public API or backend change.
Only already-resident, bounded, bias-free TP1 numeric experts use it. A miss or
other profile follows the normal fallback. Standalone `numeric_linear_into`
retains its before/after proof contract. Both numeric arithmetic profiles,
activation clamp semantics, diagnostic events, wire format, callbacks and
cancellation/deadline plumbing are unchanged.

The scope has one final-consumer admission proof. The enclosing `synchronous`
proof/drain/compute-sync/upload-sync contract is deliberately **not removed**.
For routed execution its final proof covers weighted combine, after all buckets.
For a single dense/EP worker call this still leaves an outer proof. This change
is not a claim of one CUDA event synchronization for an entire call.

## Tests and evidence limits

`numeric_scope.rs` is included by the existing CUDA bucket unit-test module and
uses its real prepared-expert fixture. It tests both arithmetic profiles:

- Actual `consumer_proof` attempt counter: the synchronous standalone-reference
  schedule asserts 8 attempts, the warm scope asserts 2 including the unchanged
  outer proof. The closure asserts zero intermediate proofs before final D2H.
- Ordered gate/up/down observations, unchanged allocation/sync/D2H counters
  between projections, and output comparison to the standalone reference.
- Error, panic and lost completion evidence after each projection.
- Real backend allocation failure at each of four preallocated operand slots.
- Scatter as the last bucket consumer, with no second bucket proof or reload.
- Workspace growth after queued scope work, with a real growth fence; a lost
  growth proof leaves old storage/allocation counters unchanged and quarantined.
- Final-consumer error/panic/Unknown, keeprefs, cache credit and allocator live
  bytes, including after owner drop and successful later sync.

The proof numbers above are **test assertions, not executed GPU results**.
Run the CUDA unit tests only under the coordinated GPU owner, e.g. the filter
`transformer::standard::cuda::expert::bucket_tests::numeric_scope` with ignored
tests enabled and one test thread. Unknown tests intentionally quarantine memory.
Then run the coordinated full40 strict and benchmark/profile lanes. Do not infer
a new M1 EventSync/token or latency result from the compile-only checks.

Validation performed without GPU execution:

- `cargo test --locked -p ferrule-model --lib`: 405 passed, 2 ignored.
- CUDA 12.4.99 / `FERRULE_CUDA_ARCH=sm_86`, `CUDA_VISIBLE_DEVICES=''`:
  `cargo check --locked -p ferrule-model --features cuda --tests` passed.
- Same toolkit/target: `cargo test --locked -p ferrule-model --features cuda
  --tests --no-run` compiled and linked all enabled model test targets,
  including these tests. The local toolkit
  lacks an unversioned libcudart link; validation-only symlinks under ignored
  `target/validation/numeric-scope-link` supplied the link names. No toolkit or
  repository build configuration was changed, and driver stubs were not used
  as runtime evidence.

- Scoped `rustfmt --check` and `git diff --check` passed. Strict Clippy was
  attempted but is not green: dependency checking reports existing backend
  `result_large_err` warnings; `--no-deps --lib` additionally finds pre-existing
  `collapsible_if` in `cuda/hybrid.rs` and `needless_borrow` in
  `cuda/kv_binding.rs`. The new scope's collapsible-if warning was fixed.
  These unrelated files were not modified.
