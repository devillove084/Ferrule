# Mixed host FP8 source retirement handoff

This change is model-only. It does not change arithmetic, rank scheduling,
collectives, CLI admission, read counters, or smaps instrumentation.

## Lifetime contract

- `HybridCudaPreparationClaim::finish(Ok(...))` publishes an owner-local completion
  observation only after the final device fence and the prepared-image check.
  `image_prepared()` alone does not permit retirement.
- The first healthy, idle Mixed invocation replaces prepared numeric host
  operands with `HybridRankStorage::NumericBinding` metadata. This happens before
  cloning the invocation Rc or submitting the forward. No-claim test/preparation
  entry points retain their sources.
- Replacement construction borrows the old image. An error/panic before the
  replacement leaves it intact. The replacement requires exclusive invocation
  ownership; there is no mutation through payload Arcs.
- Both the layer directory and the separate `bf16_deltas` directory must switch.
  The materializer is not dropped. BF16/vector/norm sources are unchanged.
- Runtime metadata owns the original binding, rank geometry, source identities,
  rectangle origins and paired read provenance. It is not a content proof or an
  uploadable artifact; host value access and re-upload/rebind fail closed.
- Legacy diagnostics explicitly retain host sources. Native diagnostics observe
  live device buffers and owned binding metadata, and do not bypass retirement.
  A late legacy callback cannot read retired numeric bytes successfully.
- This does not reduce loading peak memory, avoid checkpoint reads, or guarantee
  an RSS/PSS reduction. Routed expert prepared payloads were already temporary;
  this change targets retained attention/shared projection aliases.

## CPU checks (all GPUs hidden)

Use CUDA 12.4 with `FERRULE_CUDA_ARCH=sm_86`, `CUDA_VISIBLE_DEVICES=''` and
`NVIDIA_VISIBLE_DEVICES=void`. Supply the existing toolchain's bin directory on
PATH and its native library directories to the linker/loader. No GPU test is
selected by these commands:

```sh
cargo test -p ferrule-model --no-default-features --features cuda --lib mixed_source_retirement --offline -- --test-threads=1
cargo test -p ferrule-model --no-default-features --features cuda --lib transformer::materialize::numeric_shard::tests --offline -- --test-threads=1
cargo test -p ferrule-model --no-default-features --features cuda --lib decoder::hybrid_cuda::preparation_claim_tests --offline -- --test-threads=1
cargo check -p ferrule-model --no-default-features --features cuda --tests --offline
cargo check -p ferrule-model --no-default-features --lib --offline
```

The claim test filter above skips the ignored real-CUDA tests; never add
`--include-ignored` to the CPU run. The source-retirement tests use Weak segment
ownership checks, exercise duplicate GDN descriptor aliases, preserve provenance,
and reject raw reads/rebinding after replacement.

## GPU handoff (not executed by this change)

Only the existing authenticated launch owner may run these with a fresh GPU grant,
fresh evidence paths, pinned rebuilt artifacts, and the existing CUDA12.4/sm86 /
`CUBLAS_WORKSPACE_CONFIG=:0:0` contract. No new supervisor or coordinator is added.
Run one exact ignored selector per invocation, with `--exact --ignored --nocapture
--test-threads=1`:

Model lib claim tests (prefix `decoder::hybrid_cuda::preparation_claim_tests::`):

- `successful_transfer_never_reopens_the_device_factory`
- `unknown_final_transfer_retains_success_value_claim_and_budget`
- `clean_late_failure_drops_all_temporaries_before_refund`
- `late_failure_and_live_state_retain_claim_until_proven_clean`
- `unknown_unprepared_success_also_retains_its_value`

The first three now also assert retirement-observation state at success, Unknown,
and rollback boundaries. Keep the existing Mixed five-stage known/Unknown
preparation fault gates as well.

Rebuild and rerun the existing composition quality selectors (unchanged files):

- `full40_candidate_boundaries_two_cases_two_phases`
- `lifecycle::tiny_prepared_cancel_multistep_and_recreation`
- `lifecycle::full40_bounded_multistep_fork_recreate_cold_owner`

Run the same six lifecycle recipes from the accepted protocol, not a new recipe
set. Native boundary capture must still match all logits/routes/state/KV evidence;
its observer does not retain the host FP8 sources. Existing PASS receipts do not
certify the changed binary. Preserve them unchanged and produce fresh receipts.
Any serving quality gate/build pin refresh belongs to its launch owner, outside
this model-only patch.
