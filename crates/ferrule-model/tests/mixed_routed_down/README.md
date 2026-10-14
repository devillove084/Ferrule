# Mixed TP2 routed-down M1 internal candidate

This is **not a validated production profile** and not Marlin. Enable only by
explicitly calling `HybridMixedBlockPlan::with_routed_down_m1_candidate` on a
native GemmEx block. Full-program and preparation-claim entry points remain
native; no environment variable, model factory or legacy precision path enables
this candidate.

## Dispatch contract

- Original invocation rows **and** actual expert-down rows must both equal one.
  A multirow invocation with a one-row expert bucket is ineligible. This local
  operator contract is row-based; it does not identify semantic ForwardPhase.
- Only numeric FP8 `RoutedExpertDown`, BF16 scales, complete 128-element scale
  blocks, TP2 hidden N and local expert K are eligible. The selector authenticates
  an enrolled `BoundParameter`: complete spec (including actual Expert layer and
  expert), role, canonical ID and shared storage identity must match. The layer
  must be a resident MoE layer and the expert must be in range. Exact local
  partition and the registered source layout or its packed rebasing are required;
  matching another layer shape, reused IDs or foreign aligned origins do not
  confer eligibility. This check uses metadata only after host retirement.
- All native views remain prepared. Unsupported provider capability, role or
  geometry uses the existing native path **before submission**. Launch errors,
  poisoned scratch and Unknown are propagated, never retried as GemmEx.
- Gate/up pairing, shared/dense/GDN/QKV/attention, expert order, route IDs and
  weight bits, BF16 writeback-before-weighting, late reduction and source
  retirement retain their existing paths.
- The reused plan selects `numeric-w8a16-fused-bf16-vector`, one numeric kernel,
  zero activation/decoded-weight/global scratch and zero provider reserve.
  SM80+ is required by the native implementation. M1 is a SIMT warp reduction,
  **not** an assertion of GemmEx/HF bitwise equivalence.
- F32 output allocation/initialization and BF16 RNE writeback remain separate.
  The candidate budget adds one F32 output grant for every local expert in the
  largest routed layer, conservatively covering explicit block routes as well
  as production top-k. It does not subtract fallback workspace/provider memory.

## CPU / CUDA-hidden gates

Run with `CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void`. CUDA builds also
use explicit `FERRULE_CUDA_ARCH=sm_86` and a PATH containing CUDA's `nvcc`, so the
build need not discover a GPU.

- `cargo test -p ferrule-model --test hybrid_rank_precision`
- `cargo test -p ferrule-backend --features cuda --test numeric_w8a16`
- `cargo clippy -p ferrule-model --lib --test hybrid_rank_precision --test mixed_routed_down_frozen -- -D warnings`
- Repeat clippy with `--features cuda`.
- `cargo test -p ferrule-model --features cuda --lib candidate_tests` runs the
  metadata-only retirement test with both GPU tests ignored.
- `cargo test -p ferrule-model --features cuda --lib candidate_tests --no-run`
  compiles the actual dispatch and Unknown gates without opening CUDA.
- `cargo clippy -p ferrule-model --features cuda --tests -- -D warnings` also
  lints the library test fixture (the narrower `--lib` target does not).
- `cargo test -p ferrule-model --features cuda --test mixed_routed_down_frozen --no-run`
- CPU-only actual checkpoint budget gate:
  `cargo test -p ferrule-model --test mixed_routed_down_frozen real_tp2_candidate_header_budget_stays_within_22gib -- --ignored --exact --nocapture`
  with `FERRULE_MIXED_HEADER_DIR` set to the checkpoint directory.

The header budget gate uses 40 layers, max rows 8, two sequences, context 128,
page size 16, 64 physical pages and five state copies. It includes 128 MiB owner
margins and requires <=22 GiB, with a one-byte strict rejection test. Other
capacity configurations require their own estimate; this is not measured peak
GPU memory.

## Identity, retirement and actual-dispatch tests

`admission.rs` is included by `hybrid_rank_precision.rs`. It rejects same-role,
same-ID, same-shape bindings from the wrong layer/expert or another storage root,
non-Expert residency and foreign source origins. It checks default-off behavior,
exact capability metadata, native fallback reserves and additive budget grants.

`dispatch.rs` is a private CUDA library test module, not a factory or production
entry point. Its CPU test drops every prepared raw payload, verifies weak payload
references are dead, deletes the source file and checks that retired metadata
still authenticates the two lanes without source I/O.

The two ignored GPU tests require separate authorization and
`FERRULE_RUN_MIXED_RANK_GPU=1`, `CUBLAS_WORKSPACE_CONFIG=:0:0`:

- `retired_candidate_dispatches_only_down_and_preserves_routed_layer_words`:
  after completed upload and host retirement, run one synthetic routed MoE layer
  on each TP2 lane, compare every BF16 word with native, verify two actual down
  candidate submissions and no artifact uploads. Original M2 with M1 expert
  buckets must remain native. This is not a full decoder layer or real oracle.
- `retired_candidate_unknown_quarantines_without_native_retry_or_release`:
  inject typed Unknown only after a successful actual candidate submission;
  require stop-on-first-Unknown, no native retry, retained BF16/F32/image custody,
  quarantine and no allocation release on retry/quiesce/drop. The injection and
  submission counter are `cfg(test)` only; actual backend failures are propagated.

The authorized small run executed both GPU tests successfully; see `RESULT.md`.
The independent frozen gate failed two W8 words while native matched every word,
so promotion remains blocked. Synthetic word equality does not override that gate.
The unit-only retirement receipt is used after synchronous image preparation in
GPU tests; it does not introduce a production retirement bypass.

## Frozen small operator gate (required before promotion)

`tests/mixed_routed_down/frozen.rs` consumes an independently frozen fixture.
The root `mixed_routed_down_frozen.rs` is only an integration-test shim, preserving
the target, exact test names and environment contract for the independent oracle. It
does not regenerate expected values, load a model on CUDA, or accept tolerance
metrics as word equality. The external fixture is not supplied by this change.

Set `FERRULE_MIXED_DOWN_FIXTURE` and independently authenticated
`FERRULE_MIXED_DOWN_SHA256` (SHA256 of `SHA256SUMS`). The manifest must list and
hash `manifest.json` and all eight files below:

- `rank0.input.bf16`, `rank1.input.bf16`: one frozen local SwiGLU row
- `rank0.weight.fp8`, `rank1.weight.fp8`: original packed N x local-K raw bytes
- `rank0.scale.bf16`, `rank1.scale.bf16`: original local BF16 scale grids
- `rank0.hf.partial.bf16`, `rank1.hf.partial.bf16`: independent HF-eager local
  down-projection outputs **before** route weighting or cross-rank reduction

All word files are little-endian. `manifest.json` identifies schema
`ferrule.mixed-routed-down.tp2.v1`, role `RoutedExpertDown`, `tp: 2`, `rows: 1`,
`oracle: "hf-eager-bf16-rank-partial"`, `n`, `local_k`, `layer`, `expert` and
`checkpoint_sha256` identifying the frozen source manifest. Preserve the
checkpoint slices, input provenance and independent oracle provenance in that
manifest; an arbitrary locally computed hash is not independent acceptance.

First run the ignored CPU `frozen_tp2_down_fixture_preflight` test explicitly.
Only after CPU gates and sole-owner GPU handoff, set
`FERRULE_RUN_MIXED_RANK_GPU=1`, `CUBLAS_WORKSPACE_CONFIG=:0:0` and run only
`frozen_tp2_down_w8a16_and_native_match_every_word` with two assigned GPUs.
The gate checks both rank partials word-for-word, native control, input
immutability, output canaries and prelaunch stride-error atomicity. A mismatch
blocks promotion; no ULP, absolute/relative or aggregate tolerance substitutes.
This backend operator check still needs a candidate model-dispatch GPU check.

Before enabling a full-program candidate, also pass the existing BF16 tie tests,
W8A16 poison/capture gates, and Mixed completion-bridge/typed Unknown retention
and failure tests. CPU typed-Unknown classification is not a GPU quarantine test.
Then require full40 logits **and all 2178 boundaries**, exact routes/weights and
retirement checks. Only after these gates run lifecycle six recipes and HTTP
smoke. Do not interpret compilation, budget admission, ignored tests or this
local operator gate as production quality/performance acceptance.

## Authenticated small-run evidence

See `RESULT.md` for the pinned CUDA12.4/sm86 run and exact failure coordinates.
`small_gate.py` reuses the existing sealed worker, all-device locks, bounded
subreaper and idle checks; it requires the base Python3.12 with `-S -B`. Its
`--prepare` mode is GPU-hidden and builds only the model unit-test and frozen
integration-test binaries. `--run` additionally requires the prepared receipt
SHA and explicit sole-owner grant; it never runs a full model.

The frozen test now collects both ranks and the independent native control before
its final exact assertion. `FERRULE_MIXED_DOWN_REPORT_DIR` may point to a fresh
evidence directory for exclusive-create raw actual outputs and `exact.json`.
This does not modify the frozen expected values or permit tolerance acceptance.
`reduction_diagnose.py` is CPU-only, diagnostic-only and cannot promote a profile.
