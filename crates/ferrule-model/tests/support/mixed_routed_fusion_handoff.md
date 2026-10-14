# Routed exact auxiliary fusion — model source freeze

2026-10-11. Model implementation and GPU-hidden type/CPU gates completed.
**GPU numerical equivalence, real-path counter assertions, full40 acceptance and
performance are NOT validated by this handoff.** No binary/full-repository freeze
is claimed. Verify `mixed_routed_fusion.sha256` from the repository root and the
backend's `pair_swiglu.sha256` before building validation binaries.

## Scope and execution policy

- Changed production code: `standard/cuda/hybrid_rank/mixed.rs` and its pure
  `mixed_output.rs` helper. `hybrid_parallel/mixed_estimate.rs` retains the prior
  numeric envelope; its comment explains why savings are not spent.
- No backend/native, runtime, composition, factory, GDN, shared/dense arithmetic,
  route upload, indices/H2D, NCCL or capture implementation changes. No temporary
  environment switch or new arithmetic profile. Public signatures are unchanged.
- The existing capability is a routed gate/up pair, native BF16 GemmEx arithmetic,
  and exactly one numeric segment per source. It uses the existing prepared
  full `[M,2N,K]` plan, original source bindings, selected-expert buckets and cache.
- No diagnostic observer: `execute_bf16_pair_swiglu_view_into` for a decoded pair;
  otherwise `execute_pair_swiglu_view_into` with the original immutable FP8 gate/up.
  The result is caller-owned contiguous `[M,N]` BF16. No second SwiGLU call.
- Either native OR legacy diagnostic observer: original same-2N pair split,
  original two-rounding `swiglu_bf16`, real gate/up trace tensors. An installed
  observer whose filter rejects everything still chooses this path. No synthetic
  CPU values, post-submit GEMM replay or access to an unexported provider tail.
- Shared/dense projections and GDN do not satisfy this routed-pair capability.
  Both observer states still share the same downstream linear and layer loops.
- A decoded-cache miss chooses original-source execution BEFORE submission.
  Submission/down/completion errors propagate; no retry into old/native math.
  Existing source retirement, borrowed leases and the outer completion fence
  remain authoritative. Unknown quarantines the owner and retains backing.

## Packed down output and nonfinite contract

Old output chain per expert: BF16 down -> F32 expansion kernel -> F32 D2D copy;
then one entire-group F32 -> BF16 kernel before the original weighted combine.
New chain: each down writes a disjoint `[Me,hidden]` BF16 slice at
`route_start * hidden`, exactly `Me * hidden` words. GemmEx's original beta=0
writes the entire slice. No read/write alias with input, weights or workspace;
neighbor canaries and source retention are covered by GPU test assertions.

Original expert-ID ascending BTreeMap order, row IDs, F32 routing weights,
F32 multiplication followed by BF16 round, and F32 addition followed by BF16
round are unchanged. This is not vLLM grouped/fused MoE accumulation.

For packing alone, finite values, both signed zeros and both infinities preserve
BF16 words. This is NOT a claim of equality for all 65,536 input words: the old
F32 -> BF16 converter sets the quiet bit on NaNs; direct views preserve sNaN
words until arithmetic consumes them. NaN payload/signaling-bit equality is not
part of the new packing contract. Existing arithmetic still propagates NaNs;
full-vocabulary validation must reject nonfinite logits, not count them equal.
CPU tests exhaust non-NaN words and explicitly exercise sNaN quieting. A separate
ignored GPU test reads raw u16 words and asserts the actual old conversion's
quiet-bit behavior, without using a CPU tensor as a diagnostic observation.

## Expected counters and simultaneous grants (not measurements)

Let E be selected expert buckets, R total routes, m each bucket's rows, H hidden
width and N local intermediate width. No GEMM is removed or grouped.

- Tail fusion saves E compute kernels: split + standalone SwiGLU becomes one
  extract/SiLU-round/multiply-round tail; same full-2N GEMM remains.
- Packing removes E expansion kernels plus one group conversion, and E D2D
  copies. Combined expected compute-kernel reduction: **2E + 1**.
- Versus the new packing path with native diagnostic enabled, no-observer fusion
  alone saves E kernels and 2E caller allocations (gate/up). Tests compare this
  narrower delta with equal cache/route conditions and no diagnostic readbacks.
- Packed-output savings relative to the original chain are
  `grant(4*R*H) + sum(grant(2*m*H) + grant(4*m*H))`.
  Unobserved fusion additionally removes `sum(2*grant(2*m*N))`.
- All buckets' remaining gather/activation buffers survive the same fence;
  pair tail/provider workspace, source weights, decoded leases, routing metadata
  and final output remain accounted for. No budget reduction or relaxation,
  provider resizing or early cache/activation release is introduced.

## Validation at source freeze

All commands kept `CUDA_VISIBLE_DEVICES=''`, `NVIDIA_VISIBLE_DEVICES=void`.
CUDA configuration follows the frozen backend handoff: CUDA 12.4.99, sm_86,
`CUBLAS_WORKSPACE_CONFIG=:0:0`, tao compiler/libs, target `pair-swiglu-cuda124`.

| Gate | Result |
| --- | --- |
| CPU `--lib mixed_` | 58 passed |
| CPU `--test mixed_routed_output` | 7 passed |
| CPU `--test hybrid_mixed_layer` | 13 passed, 1 header-only ignored |
| CPU/CUDA `--test mixed_decoded_budget` | real-header test ignored; no admission claim |
| CUDA-linked `mixed_routed_output` binary, GPUs hidden | 7 passed |
| CUDA strict clippy `--features cuda --lib --tests -- -D warnings` | passed, including new ignored GPU tests |
| CPU strict clippy `--lib` plus the three targeted integration tests | passed |
| CPU strict clippy `--lib --tests -- -D warnings` | blocked by existing non-CUDA test-only `numeric_shard.rs:224 payload_liveness` dead code; not changed |
| Scoped rustfmt / whitespace | passed |
| Frozen backend manifest verification | all entries passed |
| CUDA `--lib mixed_` test build | 120-second initial build timeout; no test execution result |
| CUDA `hybrid_mixed_layer` test build | 120-second build/link timeout; no test execution result |
| GPU small/full40/QA8/timing | not run |

The validator must finish/link the CUDA unit and broader integration binaries
with an authorized longer build budget, then hash the resulting executables.
Strict clippy is type checking, not proof that GPU test binaries linked or ran.

## Small GPU selectors — sole-owner authorization required

Use `cargo test -p ferrule-model --features cuda --lib SELECTOR -- --exact
--ignored --test-threads=1 --nocapture`, with the existing authorization
`FERRULE_RUN_MIXED_RANK_GPU=1`, backend-pinned toolchain and authorized visibility.
Do NOT use the hidden-device settings when intentionally executing GPU gates.

Prefix for the four new tests:
`transformer::standard::cuda::hybrid_rank::mixed::decoded_gpu_tests::routed_fusion::`

1. `routed_fusion_noobs_and_obs_exact_with_launch_and_allocation_counts`
   - Both rank-local lanes; M1/M2/M5; native sources and cached cold/hot paths.
   - Retire host sources, remove the source fixture, then execute real routed
     buckets with reversed router slot IDs and non-BF16 F32 route weights.
   - Noobs must submit two fused tails; native observer must submit zero and
     expose two real gate/up pairs. Exact BF16 result equality, E-kernel and
     2E-allocation deltas, unchanged upload/fence counts, no D2H, idle leases.
2. `routed_pack_words_nonfinite_contract_and_removed_launches`
   - Actual old expand/copy/convert for E=2 versus direct packed words;
     E+1=3 removed pack kernels. Raw u16 readback over every source word,
     non-NaN equality and explicit old sNaN quieting.
3. `routed_fusion_unknown_retains_leases_and_never_replays`
   - Typed Unknown after the first actual fused tail; no next bucket/native
     replay; subsequent execution rejected without launches; backing retained.
4. `routed_fusion_after_down_failure_does_not_replay`
   - Ordinary failure and typed Unknown after actual down submission. Exactly
     one pair and down; ordinary failure settles leases, Unknown retains them.

Existing canary selector (without `routed_fusion::` in the prefix):
`transformer::standard::cuda::hybrid_rank::mixed::decoded_gpu_tests::routed_down_staged_slices_match_standalone_and_preserve_neighbors`

It now forces its reference through the old observed split+SwiGLU path, while
staged noobs runs fusion. M1/M5, native/cached, cold/hot; compare full middle
slice and both neighboring canaries, and assert input words remain unchanged.

Also rerun existing `decoded_physical_full_native_fallback_before_submit_exact`,
cache/source-retirement/Unknown tests, shared fixed-pin tests, and the backend
rank-pair/edge/poison selectors from `pair_swiglu_handoff.md`.

## Full40 validation: distinguish fused execution from observation

No full-layer loop or factory was copied, and no required-field capture was
removed. The existing 2178-field boundary acceptance requirement remains intact.
These existing selectors have DIFFERENT coverage after this source freeze:

| Existing test | Pair tail actually selected |
| --- | --- |
| `ferrule-composition --test tp2_mixed_execution full40_independent_logits_prefill_one_decode` | **Fused**: `full40(false)`, no observer, native source path; Hello/Capital prefill and decode, all 40 layers, independent finite full-vocabulary comparison |
| `... full40_candidate_boundaries_two_cases_two_phases` | **Split diagnostic**, not fused-tail evidence |
| `... decoded_cache::full40_cache_candidate_boundaries_two_cases_two_phases` | **Cached split diagnostic**, not cached fused-tail evidence |
| `... lifecycle::full40_bounded_multistep_fork_recreate_cold_owner` | **Split diagnostic**: recipe installs capture |
| `... lifecycle::full40_decoded_cache_bounded_multistep_fork_recreate_cold_owner` | **Cached split diagnostic**: recipe installs capture |
| Model `production_diagnostic::full_image_*_boundaries` | **Split diagnostic**, including the complete Capital forward |

The no-observer full40 selector is mandatory IN ADDITION to full-boundary gates.
Do not report a full-boundary pass as a fused-path pass. The existing named
no-observer full40 selector does not enable the decoded cache; cached noobs has
small-test coverage here, not full40 proof. A validator claiming cached fused
full40/QA8 acceptance must additionally replay the existing unobserved production
stage with its already-admitted cache configuration and pin that exact run.
Do not disable required capture inside boundary tests to manufacture this claim.

Full40 command (only under separately authorized TP2 GPU custody, existing
checkpoint/oracle pins and evidence environment):

```sh
cargo test -p ferrule-composition --features cuda --test tp2_mixed_execution full40_independent_logits_prefill_one_decode -- --exact --ignored --test-threads=1 --nocapture
```

Verify the complete qualified names with the rebuilt binary's `--list` before
running the other selectors. Existing performance summaries expose
`diagnostic_observer_enabled`; performance timing alone does not install a
numeric observer. Preserve both noobs execution evidence and diagnostic captures
as separate receipts; do not extrapolate QA8 launch or latency savings here.
