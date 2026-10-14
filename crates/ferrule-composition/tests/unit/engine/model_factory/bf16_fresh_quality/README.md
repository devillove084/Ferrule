# Bounded fresh BF16 Mixed quality gate

This test reuses `prepare_full40`, the public factory's actual retained runner,
`full_logits_step`, `StrictReport::bf16_values`, and `check_state`. It does not
change model code, the7GiB cap, original tolerance, or the full40 reference.

Inputs are frozen before inference in `INPUT.json`: three raw-completion prompts
(no chat template, no added special tokens), each <=8 prefill tokens, then8 cached
decode steps (9 next-token predictions including prefill). Greedy means F32
comparison of BF16 logits, exact max with lower token ID tie break, ignore EOS.
Prefix reuse is off. Each prompt starts with fresh KV/conv/F32SSM state.

`oracle.py` imports the maintained pinned BF16 HF initializer, loader, model,
router observer and state observer. It executes original HF functions on original
FP8/BF16 weights with F32SSM, not Ferrule outputs. It saves full vocabulary logits,
routes, layer outputs and all state for each step. PyTorch2.13.0+cu129 and
Transformers5.5.3 execute separately from Ferrule CUDA12.4. Every source/payload,
tokenizer, oracle, executable and original-reference identity is recorded.

`mixed_fresh_quality_retained_generation` first compares common HF history at
every step, including exact top1 and unchanged numerical/state bounds. Only after
ALL requested common-history cases pass does it run independent candidate feedback,
then bounded reset and fork snapshot preservation on the first case. Separate `mixed_fresh_quality_cold_owner_restart`
is blocked until the primary quality gate passes. It checks the first prompt
again in a new process/owner. These are small protocol checks, not a broad quality
benchmark or an equal-quality performance comparison.

On failure, preserve the first step's metrics and stop later prompts/free feedback/
cold restart. `FERRULE_BF16_FRESH_NARROW_DIR` is a test-only diagnostic switch for
that first QA decode: only layers14..16 are recorded, the intervening fork is
omitted, and no reference arithmetic or tolerance changes. `replay.py` runs only
native layer15 using the saved original predecessor hidden and cached KV,
validates its attention/MoE/hidden/KV against the fresh independent oracle, then
compares available actual operands and SAMEINPUT projections.

## Frozen test profiles and maintained selectors

`bf16_full40.rs` owns the checked `Full40TestConfig`; both gates use the same
`prepare_full40_with_config` and real numeric BF16 factory. The two-argument
`prepare_full40` remains a historical8 wrapper for existing diagnostic callers.
No environment override can change the capacities behind an old selector.

| Setting | `historical8` | `storage-middle64` |
| --- | --- | --- |
| Expert entries / total cache | 8 / 320 MiB | 64 / 320 MiB |
| Scratch inside total cache | 0 for original BF16, 8 MiB for tiled/sliced/Mixed | 8 MiB, Mixed only |
| Workspace / device / KV ceilings | 2 GiB / 7 GiB / 16 MiB | unchanged |
| Allocator margin | 512 MiB | unchanged |
| Context / sessions / batch rows / prefill chunk | 128 / 2 / 8 / 8 | unchanged |
| Maximum decode batch | 1 | 2 |
| Prefix reuse / native proposals / stop-at-EOS | disabled | disabled |

Storage-middle64 is checked against the frozen
`papers/arxiv/experiments/storage_demand/case-middle.json` capacities. The test
uses explicit `Bf16ModelMixedF32Accumulate` + `E4M3FnBf16Scales`, not the CLI
`model_profile` selector. Requested/effective factory options, resolved capacity,
precision, resident KV, lazy expert policy and unsharded topology are asserted
and included in `test_config` / `effective_config` / `source_lowering` reports.
This is a lowering relation, not a hash match to the historical sealed server.

Under `model_factory::dense_bf16_tests::full40::`:

- `numeric_bf16_mixed_storage_middle64_public_factory_full40_logits_routes_states_match_profile_reference`
- `fresh_quality::mixed_storage_middle64_fresh_quality_retained_generation`
- `fresh_quality::mixed_storage_middle64_fresh_quality_cold_owner_restart`

The cold storage-middle64 selector requires an explicit primary report from the
same frozen config/reference; historical8 PASS cannot admit it. Existing numeric,
fresh and capture-only selectors keep historical8 capacities and decode limit.

The numerical denominator and tolerances remain unchanged: full40 has 804
numerical metrics / 1,986,560 logits, fresh has 5,427 numerical metrics /
9,187,840 logits; `BF16_ATOL=0.00025`, `BF16_RTOL=0.012`. Full40 additionally
checks four next-token predictions; fresh checks 27 common-history predictions,
then every independently fed-back and reset prediction. Exact token checks are
recorded separately rather than changing numerical metric denominators. All
use the existing exact greedy, lower-token-ID tie policy. Frozen oracle next IDs
are also checked against their stored final-row logits before GPU construction.
These are direct-runner observations, not HTTP generated token IDs.

All numerical gates install diagnostic observers. Reports therefore say
`observer_mode=diagnostic_routes_and_states`, `ordinary_no_observer_claim=false`
and `concurrent_scheduler_claim=false`: CPU reads/state snapshots may force
synchronous or fallback paths; configured sessions2/maxdecode2 does not mean the
direct sequential test exercised simultaneous requests or ordinary fused serving.
A capture-only selector never becomes an acceptance gate.

With GPUs hidden and pinned CUDA 12.4 available, build only this package's library
test target with `cargo test --locked --offline --release -p ferrule-composition
--no-default-features --features cuda --lib full40 --no-run`. List selectors
without execution using the resulting executable and `--list full40`.
CPU-only contracts are nonignored; two explicit ignored CPU-only selectors are:

- `numeric_bf16_mixed_storage_middle64_full40_metadata_budget_matches_source_headers`
  (source headers, projection/admission and effective-plan assertions; no owner)
- `numeric_bf16_storage_middle64_reference_greedy_ids_match_frozen_logits`
  (pinned local reference artifacts only; no candidate inference)

Both CPU-only selectors require `CUDA_VISIBLE_DEVICES=''`; do not broadly use
`--ignored`, which would execute unrelated GPU selectors. Select a single exact
CPU test. New GPU selectors remain ignored and require sole-owner authorization,
`CUBLAS_WORKSPACE_CONFIG=:0:0`, pinned reference hashes and new report paths.
No historical receipts or reference artifacts are rewritten.

## Latest pinned evidence (2026-10-07)

[Prepared-attention FINAL.json](../../../../../../../target/validation/bf16-attention-model-current-20261007/FINAL.json)
records PASS for the original full40 (804 metrics) and this bounded fresh gate
(5,427 metrics, including 27 logits metrics / 9,187,840 elements). All three
common-history cases and independent free-greedy feedback passed unchanged
bounds; QA fork/reset and separate cold-owner QA replay passed. No reference
regeneration or threshold relaxation. Logits are not bit-exact; arithmetic is
incomplete within the fixed token budget. This is not broad task quality,
public-serving acceptance for that binary, or performance evidence.

See [current acceptance](../../../../../../../docs/CURRENT_ACCEPTANCE.md) for
source/binary scope and the separate earlier Hello serving receipt. Later
source changes require renewed validation; historical failures remain below.

## Historical first failure (superseded for the latest pinned evaluation)

`target/validation/bf16-fresh-quality-20261007/`:

- Independent HF completed all3 prompts x9 predictions.
- Ferrule QA prefill: all201 metrics have zero measured error, including1,241,600
  logits and all KV/conv/SSM. Predicted Tokyo, matching HF.
- First decode on token25358 at position5: top1 period matches HF, but199525/248320
  logits exceed original bounds; max error .53125.13 ordered-route ID mismatches.
- Narrow replay without fork gives identical metric records. First differing
  checked producer boundary is layer15 native BF16 `SharedExpertOutputGate`,
  M1/N1/K2048. Exact input, actual -2.046875 versus HF -2.0625; difference .015625.
  The later layer16 state/router errors are propagation, not evidence of a GDN defect.
- `frozen-shared-output-gate/` contains original BF16 input/weight/actual/expected,
  checkpoint byte provenance and hashes. `LAYER15-REPLAY.json` has5/5 exact native
  output/state validations and SAMEINPUT replay.
- No model/kernel fix was made; free-greedy, reset and cold restart were not run
  after the failed quality prerequisite. Arithmetic/Chinese Ferrule cases remain
  unexecuted. Do not report oracle-only generated text as candidate output.

The first HF child failed before model execution because peak stats were reset
before CUDA initialization; the setup ordering was fixed and its failed receipt
retained. All GPU children use the existing authenticated bounded subreaper,
not a new supervisor. Exact commands/environments are in `PREPARED.json`.

Narrow diagnostics can select a known failing case, cached step1..8 and four adjacent layers through the checked `FERRULE_BF16_FRESH_NARROW_CASE`, `_STEP`, and `_LAYER` test environment. These do not alter the oracle or acceptance bounds. Historical first-failure results above are preserved; see the scalar follow-up evidence for resumed progress.
