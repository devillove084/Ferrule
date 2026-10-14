# Model PP/EP contracts and runtime coverage ownership

```sh
cargo test -p ferrule-model --features cpu-reference --test hybrid_pp_ep --test hybrid_tp_ep
cargo test -p ferrule-runtime --features cpu-reference --lib parallel::pipeline::hybrid_tensor::tests
```

## What remains in model tests

Four hybrid layers are partitioned `[0,2)` / `[2,4)`. Each PP stage executes the
same `HybridTensorCpuSegment` used by TP1/2/4, with GDN, gated GQA, routed/shared
MoE, unequal top-2 weights and a scalar shared-output gate. The oracle is the same
checkpoint's unsplit `HybridDecoderSegment`, not copied forward math.

`hybrid_pp_ep.rs` transfers the real hidden boundary between independent local
physical transactions. It compares logits, recurrent/conv state and every GQA
history within `3e-5` during prefill and successive decode. Local commits only
seed the next numerical step; they make no all-owner publication claim. It also
checks stage/endpoint owner mismatches during preparation and transaction/session
count mismatches before any state mutation or EP dispatch.

The PP stage owners are explicitly 7 and 19. TP tests use `[7,19,31,63]` and a
nonzero leader. Endpoint owner and model layout owner must agree. There is no
synthetic dense topology, `PipelineRank { global: lane }`, or fake global proof.
Runtime rejects sparse KV meshes that do not match its authenticated topology;
these model tests do not change that admission policy.

The old `support/hybrid_pp_ep.rs` and `support/hybrid_tp_ep.rs` are deleted,
including `Program`/`RankProgram`/`PipelineStateProgram` implementations, worker
boot, manual Prepare/Execute/Ready/Install/Publish/Retire/Finish drivers, and
cross-owner fault/retry orchestration. Reusable setup now lives in
`hybrid_parallel/{model_contract,local_transaction,experts,probe}.rs`;
fixtures remain data-only. See [the model contract](hybrid_tp_ep.md).

## Cross-owner coverage transferred to runtime's existing sealed factory suite

Directory: `crates/ferrule-runtime/tests/unit/parallel/hybrid_tensor/`.
Runtime owner maintains these tests; no runtime files are changed by model cleanup.

| Removed model driver claim | Real runtime coverage |
| --- | --- |
| All owners complete and publish one PP/TP/EP step | `mod.rs::actual_pp2_tp2_ep2_matches_unsplit_owner_logits_local_state_and_kv`; `protocol.rs::all_kv_owners_ready_before_install_and_ep_owners_never_vote` |
| Late EP failure/cancellation rolls back every owner | `retry.rs::tp1_real_in_compute_cancel_and_ep_error_retry_after_journal_rollback_tp2_stays_poisoned` uses the public factory and real child compute hooks |
| Successful new-token retry after cleanup | Same `retry.rs` test proves TP1 journal rollback permits transaction 3 with a different token; TP2 must remain poisoned, not replay |
| Predecision cancellation leaves no partial state | `custody.rs::predecision_cancel_rolls_back_all_owners_and_new_step_matches_oracle` |
| Existing partial page/COW/new-page abort and state/KV restoration | `custody.rs::fork_cow_and_new_page_abort_preserve_the_source_and_physical_capacity`, plus numerical step histories; see remaining matrix below |
| Failure never partially publishes; Unknown retains custody | `custody.rs::late_failure_never_partially_publishes_and_unknown_retains_parent_custody`; `typed_unknown_retains_full_cohort_credits_new_page_and_frontier_across_cleanup_attempts` |
| Pending/Unknown cannot permit retry, release or successful shutdown | `retry.rs::singleton_unknown_or_pending_rollback_never_releases_step_admission`; `child_lifecycle.rs` child cancel/failure/Unknown and shutdown cases |
| Prewrite identity disagreement | `protocol.rs::global_identity_mismatch_is_detected_before_any_state_write_or_dispatch`; model retains direct token/hidden/generation/binding preflight checks |
| Nonzero EP source, no-EP dense, TP1/TP4 and sparse ownership | `mod.rs` optional-attachment/sparse-KV rejection, `startup.rs` dense TP1/TP4, and model sparse collective contracts |
| Schedule and singleton admission | `schedule.rs` V2 metadata translation/sealed comparison; `protocol.rs` TP1 empty-wire admission |

### Runtime migration acceptance matrix

The runtime owner should retain late **in-compute** error and cancellation cases
at both context=3 (existing partial page) and context=4 (new page), checking every
owner's state/KV/capacity, no publication, correlated child completion and no EP
KV vote. At cleanup completion TP1 may retry with a different token; TP>1 stays
poisoned. Pending/Unknown must retain cohort/page/credit/frontier custody through
repeated retry/release/shutdown attempts. The inspected `retry.rs` in-compute
matrix currently starts at context=2 (new page); COW/partial-page tests exercise
separate paths. Do not treat these as proof of every late-fault/page-shape
combination until runtime adds the full matrix.

Model retains panic -> poison/abort -> original unwind, caller-owned cleanup,
newer-transaction rejection, different owner-local generations, checked ranges
and allocation overflow tests in `hybrid_parallel/review.rs` and `state_bounds.rs`.
No GPU validation or production admission is implied; the production gate remains.
