# Pressure / fault — new observation-budget v2 study

The CPU plan does not authorize execution. A separate explicit parent GO and
paused profiling-owner handoff permit the narrow pressure entry after the exact
nineteen-file source whitelist has been tested, committed and pushed. Only this
directory is owned; production, manuscript, shared harness and historical
receipts stay unchanged. Component faults remain separate and unauthorized.

## V2 status: CPU prepared, not launched

`plan_v2/` is a **new prespecified study**, not a resumption or successful
replacement of `artifacts-go-20261015-01`. The original `plan.json`, `bindings.json`
and all historical artifacts remain byte-for-byte unchanged. New execution uses
only `plan_v2/plan.json` and `plan_v2/bindings.json`; prior results are not pooled.
The only new authorized-directory name is `artifacts-go-v2-20261015-01`.
A fresh explicit parent GO and paused-profiler handoff are still mandatory.

The stopped small point recorded three successful HTTP requests, including only
**two** exact HTTP canaries, and one censored Chinese request. All 26 worker IDs
were enqueued and the three raw eight-ID sequences matched, but the fourth HTTP
usage/terminal gate did not pass. Middle and big were unissued; no point qualified.
Neither raw IDs nor restored idle upgrade that outcome.

Source inspection found **three full validation callbacks per request**, each
rehashing workflow and candidate pins, plus pre/post NFS/PID-I/O observation.
The prior 60s aggregate included this work. Recorded non-HTTP gaps before the
last three requests were 9.5808, 9.7081 and 9.7344s; Chinese had only 6.8733s left.
A separate CPU-only probe measured 0.0902s for 88 workflow pins (94,864,532 bytes)
and 3.0401s for 2,151 candidate pins (3,182,160,745 bytes), consistent with the
structural cost. **Historical helper timings remain unobserved/null**: this
probe does not reconstruct which helper consumed each historical gap.
`plan_v2/root-analysis.json` preserves the limited counts and source-counter scope;
`plan_v2/cpu-probe.json` labels the separate probe.

## A. Prospective coupled cache pressure

Freeze `small → middle → big`, one fresh lifecycle each, **blocking only**:

| Point | Requested entries | Cache including scratch | Byte-only payload bound | Joint positive-payload bound |
|---|---:|---:|---:|---:|
| small | 8 | 64 MiB | 18 | 8 |
| middle | 64 | 320 MiB | 103 | 64 |
| big | 128 | 640 MiB | 210 | 128 |

The existing actual-header receipt reports `P = 3,146,112` bytes per maximum
whole expert: three FP8 matrices plus paired BF16 scales. Calculate
`min(entries, floor((cache_bytes - 8 MiB) / P))`. All three are entry-bound under
this arithmetic. This is **two jointly varied knobs**, not independent LRU or
byte-budget sensitivity, occupancy, a physical capacity frontier, or a claim
that 320 MiB holds the working set. The source has 40 × 256 = 10,240 routed
expert IDs; bounds represent 0.078125%, 0.625%, and 1.25% of that count.
Actual selected unique IDs are unavailable. The six possible source extents per
whole expert do not establish six physical disk reads or an observed extent count.

Keep device7GiB, host64MiB **including** metadata8MiB/stack2MiB, scratch8MiB,
KV16MiB, workspace2GiB, margin512MiB, context128, sequences2, batch/prefill8,
cohort1/0, TP1/PP1/EP1, lazy experts, resident KV, NVTX off and existing trace
settings. Host retention zero is not page-cache zero. Host accounting excludes
OS/TLS/allocator overhead; do not claim a physical host cap.

At every point use exactly `hello2`, `canary-qa`, `canary-arithmetic`,
`canary-chinese` from the maintained schedule. Canonical references are checked
against `../canaries.json`, not invented answers (including the arithmetic
oracle's actual continuation). Hello2 is length/usage smoke. Each canary must
use the full 40-layer model, all eight generated IDs, exact text/usage and the
maintained real HTTP/worker-enqueue-ID join. No extra QA, warmups or retries.
Denominators: **3 lifecycles, 12 requests, 9 exact eight-ID canaries, 78 planned
completion tokens (72 oracle-covered)**. Previously measured middle five pairs
are separate history; the middle pressure point must be fresh too.

### Time and quality boundaries

Pressure cap **900 s = 3 × 240 s + 180 s study cleanup reserve**. V2 allocates
**60 s startup/preflight + 120 s requests/observations + 15 s bookkeeping +
45 s cleanup = 240 s**. Both coordinator/worker pre-idles, authentication and
candidate checks are **inside startup**, not extra time added after launch.
The sealed old pressure plan was startup75/HTTP15/aggregate60, not the older
shared protocol's startup120/HTTP60. V2 keeps HTTP15 unchanged. Four HTTP caps
use60s; the other60s of the aggregate is explicitly for validation, metadata,
Ready maps, serial preparation and pre/post observation. This is a finite
allocation, not a promise that any cache state or future startup will fit.

All offsets share the point's original monotonic origin:

| Phase completion | Absolute offset / cap |
|---|---:|
| Startup, including preflight and pre-idles | 60s |
| Request phase, including Ready checks/maps and all four slots | 180s, also at most120s from phase entry |
| Post-request maps/checks/bookkeeping | 195s, also at most15s |
| Worker shutdown/observer finish/closure | 215s, also at most20s |
| Worker finalization | 219s, also at most4s |
| Supervisor own-child cancellation/reap reserve | 230s |
| Three-sample post-idle | 235s, also at most5s |
| Coordinator finalization / point outer cutoff | 240s |

The45s cleanup envelope is20 + 4 + 11 + 5 + 5; it does **not** sit outside240s.
The supervisor's fixed219s absolute cutoff leaves11s for its maintained
8s wait,0.1s cancel send and2s direct-helper reap. Unknown/lost custody still
fails closed. A bounded phase can time out during observation or cleanup;
no shortened observation waives identity, three samples, ECHILD or quality.
The separate180s study reserve remains unspent by point admission.

Admission still requires a whole240s point plus180s reserve. No clocks slide on
progress; no unused startup allowance enlarges the120s request phase. Readiness
or validation after its cutoff is rejected even if another phase has room.
Pre-dispatch expiry is unissued; issued but incomplete evidence is censored.
Post-HTTP observer expiry cannot leave an eligible success. Stop on the first
failed gate/unknown custody/deadline, keep every denominator, and suppress any
incomplete three-point comparison. No retry, replacement, adaptive cap or extra
request is authorized. Cold-start/cache state remains unknown.

CLI86 (`86c58a0`, CLI SHA `7cc8e0ff…e607a01`, source digest
`10183d24…dc8188`) has **recorded 9/9 candidate gates**, bound in `plan_v2/plan.json` and
`plan_v2/bindings.json`. Those gates cover the existing middle configuration, including
full40/fresh/ordinary/cold numerical checks. Cache changes do not change the
selected kernel/math, but cannot inherit full-logit/route/state PASS. New-point
full numerical quality stays **unverified** even after visible eight-ID PASS.
`Full40TestConfig` exposes named Historical8/StorageMiddle64 variants, not
arbitrary caps. Parent has accepted the narrow visible scope **for coupled budget
sensitivity only**, not frontier quality/SOTA or all-cap numerical equivalence.
No production matrix or test-only numerical configuration hook is added here.

### Private four-request adapter and separately authorized command entry

`subset_adapter.py` implements the parent-approved budget variant. It validates
the exact compiled CLI86/source archive, unchanged prior raw nine-gate producers,
precision/references and runtime pins. It never invokes historical source-cut or
current-HEAD acceptance and never recaptures a stale source/generation. The prior
middle case remains middle-only evidence; small/big receive no numerical PASS.

The maintained `execution_v2.serial_requests` and `run_pilot.worker` are compiled
into **private globals**, with pinned `== 12` count expressions specialized to
`== 4`, plus pressure-only timing and absolute phase-cap wrappers. The shared
validator and all shared files stay unchanged.
The private schedule is exactly Hello2 plus the three canonical canaries. Whole-log
validation requires all four HTTP/worker IDs, exactly26 enqueued tokens, exact
admission/terminal order, full usage/text checks and no fifth request, including
an admission that produced no tokens. No empty or partial success is possible.

The worker retains maintained authentication, all-eight lock validation, capture,
observer, HTTP client, serial journaling/deadlines, closure and positive role-based
maps at Ready/after requests (the 7d map-evidence path). Only its pressure contract
schema is new; launch-guard/subreaper/custody schemas are unchanged. `next_point`
reuses the maintained no-replay coordinator selector. The private `idle` adapter
shortens the existing 15s observer to the plan's absolute 10s pre-idle envelope;
it retains all three consecutive samples, immediate memory/app rejection and
post-ECHILD identity checks. The shared observer is unchanged; v2 explicitly
budgets both pre-idles within startup and a5s post-idle within cleanup.
A parent coordinator must use this bounded helper, not `pilot.idle`.

The `worker` hook enforces and persists `point_assessment`: actual per-point source
options, positive verified reads, bounded cache charges and the full four-request
join. `lifecycle_assessment` combines it with the maintained authenticated drain,
worker-exit, closure and post-idle gates; point success alone cannot admit another
lifecycle. Invocation deltas stay separate from final owner gauges/peaks/cumulative
read statistics. The command entry delegates one point at a time to the maintained coordinator.

CPU preparation checks metadata/payload formulas against the pinned existing
actual-header receipt with the same fixed limits and only the two changed caps.
That is **not a newly executed all-cap CLI lowering test**. The maintained ignored
actual-header CLI selector reads only the frozen middle literal argv; its helper
has no arbitrary-cap argument or `--dryrun`. It was not modified or rerun here.
New exact-cap source/profile/options admission occurs in real server startup and
is checked before dispatch. Runtime rejection remains rejection. Source-plan caps,
requested caps, observed occupancy/charges and physically admitted caps remain
separate; unreported physical caps stay `null`.

`check` stays CPU-only. `start --directory <new-private-directory>
--sole-gpu-owner-granted --profiling-owner-paused` requires both explicit grants,
checks that the exact nineteen source/preparation files are committed and pushed, and writes
read-only exclusive GO, case-variant, precise build-review and registration
receipts. It starts one absolute900s clock. The old stopped study is retained,
never resumed; a second v2 GO or any unrecognized intervening study is rejected.

`next --directory <same-directory>` calls the pinned maintained finite-arm
coordinator with private entry/schema/count/budget/assessment substitutions.
It issues at most one point and retains the maintained all-eight locks, passive
pre-idle, baseline, launch authentication, owned subreaper, shutdown and ECHILD
post-idle checks. No new process/request loop is introduced. `--_worker` is only
the authenticated maintained supervisor target. `finalize` retains every planned
request denominator, including unissued points after STOP. The failure policy is
**STOP on the first error**, including known clean admission rejection, with no
retry, replacement, or selective comparison. Even if physical idle is restored,
a failed point cannot reopen admission. Private raw receipts are never staged.

The outer coordinator applies the pressure-only composable absolute alarm during
owned point work. Nested alarms take the earlier cutoff and restore only the
parent's *remaining* time; they cannot reset an outer budget. A timeout preserves failure/unknown evidence; it cannot authorize
a new run. The CPU plan remains non-authorizing; only the new immutable GO
authorizes this scope. Component fault selectors are never dispatched here.

`phase_timing.py` records named wall/thread-CPU intervals for metadata, workflow
pins, candidate pins, receipt hashes, source/custody checks, Ready/after maps,
NFS, PID I/O, HTTP, token join/source counters, pre/post observers and shutdown.
The maintained observer's private method records concurrent PSS and process
inspection spans. JSONL start/end records survive interrupted phases; absent
measurements stay `null`. Nested and concurrent intervals are **inclusive and
non-additive**, not extra request E2E or reconstructed historical overhead.
Only fixed phase names/times/status are recorded, never arguments, secrets,
environment, pin filenames or exception text. All existing rehashes and gates
remain in place: **no cache, stat shortcut, observer skip or validation bypass**.

Report requested, logical bound, and **effective admitted** caps separately;
reader bytes/calls/source checks, hits, evictions, uploads/H2D and observed source
extents separately. Missing counts, misses, hit rate, working-set unique IDs and
NAS backend bytes stay `null`. Invocation deltas may sum; owner gauges/peaks and
cumulative pipeline stats must not. No storage-necessity promise.

## B. Actual-path fault evidence, not integrated serving E5

`plan.json` gives exact fully qualified selectors and sources. Minimum future
GPU component pair (not run):

1. `transformer::standard::cuda::bf16_chunk::tests::chunk_clean_failure_retains_reusable_workspace_and_releases_operands`
   — actual Mixed numeric-owner chunk submission, clean error, operands released,
   reusable workspace, allocator baseline after drop/trim.
2. `transformer::forward::tests::cuda::hybrid_unknown_finish_keeps_transaction_pins_despite_independent_kv_fence`
   — actual tiny F32 runner, lost finish proof, retained transaction/KV/state,
   unchanged old topology/core despite independent fences; no publish/release.

Both are `ferrule-model --lib --features cuda`, ignored, ordinal0, single-threaded.
Their CUDA libtest executable is **not one of the qualified CLI/constructor/
composition artifacts**, and no current same-cut executable binding is asserted.
Parent must obtain/pin a matching artifact and exact `--list --ignored` receipt
before execution. Each future selector gets at most120s; independent B cap420s
includes180s cleanup, no retries. The second is a synthetic F32 component, not a
35B Mixed serving gate. Do not disguise that mismatch.

Also inspected, not added to the GPU run budget: `numeric_scope` projection
failure/Unknown and final-consumer retention selectors. They cover tiny numeric
BF16/F32/W8A16 expert owners, not a committed serving frontier. `TestHook` phases
in checkpoint reads and consumer-proof flags are private `cfg(test)` capabilities,
not public serving flags. No current CLI integrated fault gate was found.
**Required serving evidence must use the actual 35B numeric-FP8 Mixed owner.**
Unsupported SM103/FP4, wrong provider, or wrong executable is a blocked result,
never permission to substitute FP4 or another engine.

If parent requires integrated E5: only a private capability on the maintained
Mixed factory owner, exact post-submit/lost-proof injection, shared read/image/
transaction/request identity, and observation-only existing charge/frontier
snapshots are needed. Any production change needs fresh artifact/source/numeric
qualification. Do not build a new fault framework. Quarantine is a subset of
charged bytes, not an additional charge; snapshots do not poll, join, drain,
heal Unknown, or prove release.

## CPU validation and commands

From repository root, with GPUs hidden:

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none FERRULE_PROFILE_NVTX=0 /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/pressure_fault_v1 -p 'test_*.py' -v
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none FERRULE_PROFILE_NVTX=0 /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/pressure_fault_v1/prepare.py check
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none FERRULE_PROFILE_NVTX=0 /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/pressure_fault_v1/prepare.py plan
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none FERRULE_PROFILE_NVTX=0 /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/pressure_fault_v1/subset_adapter.py check
```

Actual worker test (180s command bound; no CUDA features):

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none FERRULE_PROFILE_NVTX=0 FERRULE_CUDA_ARCH=sm_86 RUSTUP_TOOLCHAIN=nightly-2026-08-01 RUSTUP_AUTO_INSTALL=0 CARGO_BUILD_JOBS=2 cargo test --offline --locked -p ferrule-model --no-default-features --lib checkpoint::read_pipeline::tests::cancellation_after_start_does_not_release_running_read_or_pins -- --exact --nocapture --test-threads=1
```

See `VALIDATION.md` for executed CPU selectors and limitations; `plan_v2/bindings.json`
pins inspected sources and existing receipts, excluding the active profiler.
`prepare.py` and `subset_adapter.py check` check/print only, never reseal or write
results. `plan_v2/bindings.json` seals the new budget variant and source inventory;
`STAGE_FILES.txt` is the exact nineteen-file staging whitelist (not a directory glob).
Only the explicit source whitelist may be staged; private execution directories,
raw maps/environments, candidates and binaries must be excluded. No GPU results
are implied by this source document; new private receipts carry actual outcomes.


## Future execution commands — NOT launched

Only after the parent issues a **new GO** and the profiling owner explicitly
hands over GPU exclusivity, from this directory with the committed/pushed v2
sources, use the maintained interpreter below. Do not start the900s clock while
waiting for the handoff.

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B subset_adapter.py start --directory artifacts-go-v2-20261015-01 --sole-gpu-owner-granted --profiling-owner-paused
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B subset_adapter.py next --directory artifacts-go-v2-20261015-01
```

`next` issues **one** point. Inspect its assessment before issuing the identical
`next` command for middle, then big. STOP forbids further points; do not use a
blind shell loop or retry a failed command. After all three or permanent STOP:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B subset_adapter.py finalize --directory artifacts-go-v2-20261015-01
```

Component fault/GPU profiler work is not part of this authorization.
