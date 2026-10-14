# Mixed BF16 chunk integration

Only the existing `Bf16ModelMixedF32Accumulate` schedule selects chunk GDN for
fresh-history or multi-row **sequence cohorts**. A one-row cohort with previous
history uses cached recurrence, including a one-token continuation labeled
Prefill. Fresh one-row cohorts use `Bf16ChunkPlan::new_prefill`, not the default
one-row delegation constructor. Other profiles are unchanged.

The exact pinned HF condition is `cache_params is not None and
cache_params.has_previous_state(layer_idx) and seq_len == 1`. `has_previous_state`
is distinct from allocated/initialized SSM storage. Ferrule derives this from the
validated `CudaBf16GatedDeltaState.position != 0`, not buffer values or batch mode.
Allocation/reset creates position0; checkout/fork copies it; successful work
advances it; abort discards the working state. No new unsnapshotted history flag.

Scope: HF itself resets initial recurrent state for a multirow call even with a
previous cache. Ferrule's existing segmented-prefill contract carries it; this
change preserves that existing continuation policy rather than claiming universal
HF multirow-cache equivalence.

`HybridMemoryPlan::bf16_chunk_workspace_bytes` takes the maximum checked backend
plan over GDN layers at the admitted row capacity, including capacity one with
no previous history. Its allocator grant is charged
once to the operation workspace, not per layer, sequence, expert cache, or state
copy. The existing preparation claim validates this envelope before one persistent
F32 buffer is allocated. Exact prefixes support smaller geometries on the same
ordered stream without growing the allocation or overlapping concurrent work.

All cohort plans are validated before convolution mutates working state. The
existing synchronous operator boundary retains the scratch root and holds typed
operands/state/output through error or unwind. Known completion releases temporary
holds, not image scratch. Lost completion permanently poisons the operator and
hybrid device; Drop retains scratch and operands with the existing Unknown custody.
The existing journal still owns state publication, abort, fork, and COW.

## Tests

- `plan.rs` / `history.rs`: profiles, history/cohort selection, fresh M1 execution
  identity/full scratch, cursor commit/abort/fork/reset/restore and underbudget rejection;
  CPU-only.
- `tests/hybrid_bf16/chunk/budget.rs`: shared maximal allocator grant, unchanged
  state/weights/KV and single-row admission; CPU-only.
- Composition `bf16_chunk/admission.rs`: full checkpoint headers only; exact
  workspace boundary and device-underbudget rejection, unchanged 7 GiB cap and
  512 MiB allocator margin. Requires hidden GPUs and explicit checkpoint path.
- `lifecycle.rs`: ignored GPU tests for clean post-submission failure, lost
  completion retention/reuse rejection, and allocation failure before mutation.
  Lost-proof injection is not a claim of testing a native mid-kernel fault.
- Existing Mixed tiny lifecycle exercises real runner/journal decode, abort, COW,
  fork/restore and now multi-row chunk GDN against the unchanged tiny oracle.

GPU tests require explicit sole ownership. CPU checking/compilation does not
establish GPU lifecycle or full-model numerical acceptance.

## M1 history validation readiness (GPU not run)

- `m1_submission.rs`: original actual layer9 convolved inputs, fresh output/state
  versus independent HF chunk, cached arithmetic on the same diagnostic operands,
  fresh/cached/fresh workspace pointer reuse; 15/1/15 launches and no launch allocation.
  The cached branch with zero initial state is explicitly a schedule-isolation
  test, not a fabricated actual cached-decode oracle. Supervisor must pin every
  file in `FERRULE_BF16_M1_FIXTURE` to the retained fixture manifest.
- `tests/hybrid_bf16/chunk/history_gpu.rs`: real runner/journal fresh-M1 abort/retry,
  first commit, exact-frontier fork, COW, prior-state Prefill versus Decode labels,
  reset and fresh replay. No oracle is synthesized: branch equality is tested.
- Existing Mixed tiny HF acceptance separately covers actual cached-decode state,
  projections, rollback/fork and KV restore against unchanged independent expected.
- Composition admission checks capacity1 full chunk allocation, exact workspace
  boundary, device-underbudget rejection, and the unchanged capacity8 7GiB cap.

All cohort plans are derived before any convolution submission, using the typed
working state's validated position. Independent fresh/cached cohorts in a packed
batch do not share a history decision. GPU tests remain opt-in and require a new
ownership grant; no GPU or full40 execution during CPU integration.
