# Internal decoded-BF16 cache qualification

## Current dedicated-uninitialized arena closeout (2026-10-10)

`runs/decoded-arena-uninitialized-20261010-01/HANDOFF.md` records **23 fresh GPU
gates** (backend5/model11/full40+six), **37 calls / 40293 exact metrics**, 33 exact
CPU selectors and 246 paper CPU tests. No old arena03/backend PASS was inherited.
The allocation-only root is retained before initialization; new view-fault,
first-fixed-failure and uninitialized-slot guards passed. Actual dedicated root,
current/peak and driver growth are each 3 GiB per owner; the root is charged once.
Final eight-card idle, ECHILD/drain and lock release are recorded. This is internal
numerical qualification for a new CPU release build, not public performance.

## Previous zero-initializing root-arena closeout (2026-10-10)

`runs/decoded-arena-fresh-20261010-03/HANDOFF.md` and its hashed
`acceptance-summary.json` record **37 calls / 40293 exact metrics, zero failures**.
All eight model gates plus full40/six recipes ran fresh; three unchanged backend
gates are explicitly source/tool/binary-bound reuse, not fresh execution.
The actual 32 owner snapshots count one 3 GiB root, with 1340 MiB pins and
1732 MiB ready free slots; whole-device sampled peak was 21.1044921875 GiB/card
under 22 GiB. Hot-fit zero deltas are model-owned GPU assertions, not a
full-model no-churn or speedup claim. Final eight-card idle/drain/lock release
is recorded. New serving release, smoke and matched-4GiB A/B remain pending.

## Historical segmented-cache qualification and protocol

The material below describes the retained earlier candidate. Its old source,
physical-pool accounting and release state are not transferred to the arena.

Status: fresh numerical qualification passed in
`runs/decoded-cache-frozen-pair-20261010-02/acceptance-summary.json`.
Performance and public E2E acceptance remain unclaimed; cache-only physical
counter export is incomplete. Public factories stay default off.
No W8 candidate, new factory, CLI flag, host-offload cache, or second completion
authority is introduced.

## Exact internal path

`ParallelExecutionPlan::internal_mixed_execution_inputs` calls the model's
`with_decoded_weight_cache` with physical pages, requested grant, actual owner
card capacity and checked context+allocator margins. Both resulting programs
remain in `MixedTensorExecutionPlan`. `resource_requirements_from_mixed_inputs`
uses those exact programs rather than rebuilding native/default-off plans.
`start_internal_with_decoded_cache` carries them through the enlarged report,
pre-open claim check and runtime `prepare_mixed_with_trace`. Trace reservation
is still added once and re-admitted against complete device capacities.

Existing `start_internal_with_trace` passes `None`; public callers are unchanged.
Cache-on full40 and lifecycle tests pass `Some(3 << 30)` explicitly. Recreate
passes the same configuration; fork stays in its original prepared owners.
The trace's `owner_binding` records model-admitted cache capacity/pins/working
set for every owner and epoch. Default-off trace bindings are unchanged.

## CPU selectors

Binary: `ferrule-composition --test tp2_mixed_execution --features cuda`.

- `decoded_cache::native_default_off_and_cache_report_claim_share_exact_grant`
- `decoded_cache::cache_rejects_low_budget_capacity_and_configuration_drift`
- `decoded_cache::cache_capacity_follows_device_placement_not_lane_number`
- `decoded_cache::full40_cache_header_ledger` (ignored; explicit real headers)
- `decoded_cache::lifecycle_cache_header_ledger` (ignored; three sessions)

The first three use the existing tiny BF16 fixture and certify plumbing, not
numeric cache hits. Real-header gates assert 64 physical slots, 1340 MiB fixed
pins, 768 MiB conservative whole-layer routed working set and 20,480 routed
descriptors per owner. Descriptor inventory is not allocated residency.

Observed header-only totals including 3 GiB cache, 128 MiB owner margins and
8 MiB trace reserve, per rank:

| Schedule | Total grants | Remaining under 22 GiB |
| --- | ---: | ---: |
| Full40, two sessions / five state copies | 22,584,094,720 | 1,038,225,408 |
| Lifecycle, three sessions / seven state copies | 22,648,975,360 | 973,344,768 |

These are managed admission numbers, not observed GPU memory peaks.

## Maintained paper-local entry

Use `/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B` to invoke
`papers/arxiv/experiments/tp2_resident/decoded_cache_gate.py`:

- `plan`: CPU-only requirements; no directories, GPU queries or launch.
- `prepare --id ID`: fresh paper-local run; CUDA12.4/sm_86 hidden build,
  exact selector listing and five CPU gates, independent staged test binary,
  source/tool/workflow pins. Compile and each CPU step are bounded to 180s;
  failure/timeout has no automatic retry. The shared Cargo cache is used, not
  a new target executable script. Source changes during preparation reject.
- `run --id ID --gate GATE --small-qualification PATH
  --sole-gpu-owner-granted --owner-note NOTE`: fresh explicit sole-owner grant;
  no GPU action before source/pin/small-gate validation. Reuses existing locks,
  `launch_guard`, supervisor/subreaper and `observer`. No alternative supervisor.
- `compare --id ID --gate GATE`: CPU-only native capture canonicalization and
  frozen independent comparison. Uses the original mapper/comparator functions;
  `decoded_cache_compare.py` changes only the allowed output location to this
  experiment. Oracle protocol/tensors/tolerances remain untouched. Outputs are
  separate from raw evidence and reference bundles, always exclusive-create.

Each lifecycle recipe requires the previous independent exact comparison.
The order is full40, greedy4, fork2, canary2, recreate1, cold-owner1-01,
cold-owner1-02. Full40 requires **2178 exact fields per hello/capital case**;
lifecycle requires all 1089 fields per call (10/6/5/4/4/4 calls).
A libtest pass alone is not independent numerical acceptance.

## Legacy external small envelope (not used by the frozen pair run)

The v2 runner uses the six actual source-pinned selectors listed below, not
this earlier external-envelope proposal. The retained standalone validator
expresses 19 coverage names: single/pair x M1/M5 x cold/hot/eviction-reload/
after-retire, plus known-failure/unknown-retains/borrowed-retains. Its presence
is not evidence that an external envelope or raw small-word bundle was produced.

Envelope schema: `ferrule.paper.decoded-cache-small.v1`, exact current `sources`,
`artifact_pins`, and `cases` keyed by those names. Each case contains `selector`
and `{path,sha256}` records for `binary`, `preparation`, `supervisor`, `test_exit`,
`evidence`. The preparation must bind the same `sources` and contain the exact
binary record in `binaries`. Evidence must bind its `coverage` name,
`binary_sha256`, `source_digest` (the shared contract's digest) and authenticated
supervisor `launch_id`; old receipts cannot simply be relabelled.
Supervisor must be authenticated/proven-drained successful, test exit must show
one exact passing selector, and evidence must name both owners and `passed`.
Numeric evidence has two `native_pairs` records (`owner`, pinned `actual` and
`native` raw BF16 word files); the runner hashes and compares nonempty equal-size
word artifacts itself, never merely accepts a tolerance-based pass flag.
`resources` is a pinned JSON with both owner ledgers: cache peak <= grant <=
3 GiB, complete total including margins/trace <=22 GiB, and zero **new artifact**
H2D bytes (ordinary activation/collective transfers are not forbidden).
All referenced artifacts must be covered by the seal's pin inventory.

Healthy native completion is required before reuse/eviction/refund. Unknown,
partial decode, surviving borrows and panic require retained/quarantined roots;
no native retry after a submitted unknown operation. After-retire must not
reopen checkpoint payloads or reconstruct artifacts via host DMA.

The old oracle bridge CLI only permits outputs below its own oracle directory.
The new output-location adapter deliberately uses its mapping/comparison APIs
instead: fresh canonical and comparison files stay under `tp2_resident/runs`,
without modifying oracle files or broadening the frozen input/protocol checks.

## Performance and public E2E remain separate

`performance --id ID --phase-evidence PATH` requires all seven fresh independent
comparisons and matching source pins. Its `validate_phase_decode` requires
complete, GPU-correlated per-rank prefill/decode
launch counts, strict reductions against matched native and pinned raw traces.
Managed cache miss counters and the historical shape audit cannot establish
that gate. Historical shape audit has no argument/pointer reuse evidence or
complete phase split. No measured speedup/peak memory claim is made here.

Only after fresh small gates, full40, all six recipes and performance evidence
may a newly pinned release/admission enter existing `paired.py` serving smoke
and paired E2E. This internal-only plumbing does **not** enable cache on public
serving. A public-off run must never be relabelled cache-on E2E.

## Frozen pair/pool/claim follow-up (v2 preparation)

The existing runner now builds and independently stages backend/model library
test executables as well as the composition test executable. Each build has a
240-second bound; CPU tests retain 180 seconds. It runs the three backend
host-only preflight selectors and the two model descriptor/full-claim selectors
in addition to the five composition CPU gates. Source inventory must remain
unchanged throughout. The old `0911...` binary/v1 preparation is not accepted.

The authorized GPU sequence is now directly bound to the six actual selectors
in `SMALL_GATES`: backend exact, backend rejection checks, then model
cold/hot/reload, fixed pins/shared separation, Unknown/panic, and physical-full
native fallback. Each receives its own authenticated, bounded one-test process,
resource samples and before/after all-device idle proof. Full40 refuses to start
unless all six current-binary runs passed; an encountered failure writes
`STOP.json` and prevents further GPU gates in that preparation. No placeholder
19-case envelope is needed for these source-pinned assertion-based tests.

Backend and model Unknown/panic selectors use device 0 only; their observer
active-owner roster is one device while idle checks and exclusive ownership
still cover the same authorized TP2 pair. Other model selectors use both ranks.
Test assertions prove their stated invariants but do not export every physical
cache value. Any absent numeric snapshot is explicitly unavailable, not derived
from logical grants or fabricated as measured bytes.

## Visibility correction and fresh numerical closeout

`execution_devices(gate)` is the single source for the sealed selected roster,
CUDA visibility, NVIDIA visibility and observer roster. Backend selectors and
model Unknown/panic see only GPU0. Three model selectors explicitly iterate
`lane in 0..2`, so they and composition see/select both authorized GPUs.
`validate_execution_devices` rejects any drift before launch. Sixteen CPU
runner tests passed, including roster/env equality and the unchanged 1 MiB
nonselected-device delta tolerance.

`reseal --id NEW --from-id OLD` supports workflow-only repairs without rebuilding
production: exact production/test source pins, tool inventory, staged binary
hashes and all old CPU/build receipts must still verify. It copies independent
binaries, pins the changed workflow, runs CPU runner tests and inherits **no GPU
passes**. Old failed receipts and their executed runner snapshot remain intact.

The fresh `decoded-cache-frozen-pair-20261010-02` run completed:

- Two backend + four model exact small selectors, one test per authenticated run.
- Cache-on full40, hello/capital each 2178 exact fields.
- Cache-on greedy4/fork2/canary2/recreate1/cold-owner1-01/cold-owner1-02 with the
  unchanged recipes; recreate has cache-enabled bindings in both epochs and
  cold owners have different authenticated process identities.
- Total: 13 fresh GPU gates, 37 full-model calls, 40293 comparison metrics,
  all exact, max_abs=0, failures=0. No old numeric pass was inherited.
- Sampled device peaks: 21,117,272,064 bytes on each card, below 22 GiB.
  Owner allocator maximum reserved: 20,831,010,816; maximum live-grant peak:
  20,823,739,136 bytes. These are actual owner allocator observations, not
  cache-only attribution. Raw log hashes/line numbers are retained in
  `allocator-observations-v2.json`; ANSI stripping fixed an empty CPU parser
  attempt, which is preserved separately and not accepted as observations.
- Final eight-card idle verified; both local locks reacquired then released.

Even with only GPU0 visible, nonselected GPUs had up to 1 MiB sampled positive
delta (GPU1, and GPU2 in backend preflight). This is preserved in each resource
receipt, was within the unchanged tolerance, and is **not** attributed to our
process from sampling alone. The previous two-visible/one-selected failure at
3 MiB remains a failed run in `...-01`.

Physical pool, no-retry quarantine, warm launch delta and no cache artifact DMA
invariants passed the model-owned small assertions. Their numeric cache-only
pin/active/pooled/slack/live/reserved/max snapshots were not exported. Summary
fields are null with an explicit limitation; logical admission numbers and
whole-owner allocator metrics are not substituted. Thus the summary is
`numerical_quality_pass_observability_partial`, not performance/public E2E
promotion. GPU profiling and the public cache switch remain off.
