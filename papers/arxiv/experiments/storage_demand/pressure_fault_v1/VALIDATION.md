# Observation-budget v2 — CPU validation and handoff

Status: **new study CPU-prepared; NOT launched**. Profiling retains GPU
exclusivity. No GPU query, lock, serving process, capture or fault run was issued
in this change. No production/shared protocol/manuscript/historical artifact was
edited. Parent explicitly authorized this separate source/test/preparation fix
and its scoped commit/push; the older authorization notes below are historical.

## Root finding and evidence boundary

The sealed pressure v1 used startup75 / HTTP15 / aggregate60. The shared paired
protocol's120 /60 startup/HTTP values were not its pressure settings. Three
callbacks per request rehash both pin sets; there are12 callbacks in the four-slot
loop plus a final worker check. NFS, PID I/O, serial preparation and callback work
all consume the aggregate. Historical gaps are9.5808/9.7081/9.7344s, leaving the
Chinese HTTP request6.8733s. Its raw eight-ID match does not supply missing HTTP
usage. Three HTTP successes include only two exact HTTP oracle passes; there
were26 raw enqueued IDs, one censored slot, eight unissued slots, zero qualified
points and no valid three-point comparison. These results are not upgraded.

A distinct CPU-only probe measured workflow pin hashing0.0902s (88 files,
94,864,532 bytes) and candidate pins3.0401s (2,151 files,3,182,160,745 bytes).
This supports the structural diagnosis, but historical per-helper durations
remain **null/unobserved**. No assumed hash-time subtraction or reconstructed
measurement is used. The v2 timing ledger will measure named intervals directly.

## Budget proof

One point:60s startup/preflight +120s request/observation +15s bookkeeping +45s
cleanup =240s. Three points plus180s reserve =900s. Both10s pre-idles and
validation/authentication are inside the startup phase, not additive. HTTP15 is
unchanged;4 ×15=60s leaves60s for the required validation/observation work.

Absolute offsets from point origin:60,180,195,215,219,230,235,240s. Cleanup is
20s worker shutdown +4s worker finalization +11s supervisor cancellation/reap
+5s post-idle +5s coordinator finalization. The supervisor cutoff reserves its
maintained8s wait +0.1s cancel send +2s direct-helper reap. Startup/cache state is
unknown; slow startup or cleanup remains rejection/unknown, not a waiver or an
extension. A timed-out observer cannot leave an eligible request success.

The new plan is not a performance qualification or a guarantee of completion.
All source checks, full per-callback rehashes, role-library maps, PSS observer,
three-sample all-eight idle, launch authentication, own-child custody, ECHILD,
full HTTP text/usage/token joins and source counters remain required. No stat/hash
cache, warmup, retry, replacement, quality relaxation or extra request was added.
Nine historical numerical gates remain **middle64/320MiB only**; new-point full
numerical quality is unverified.

## Tests run in this change

Passed **63/63 owned CPU tests** and **33/33 unchanged maintained execution tests**.
Tests include fake-clock slow callbacks, pre/post observer budget consumption,
120s composition versus a censored60s CPU fixture, earlier target caps, nested
alarm restoration without sliding, real blocking-callback interruption, retained
supervisor cleanup receipts, and the actual maintained worker's owned stop/reap
path under timeout. Subprocess/HTTP/GPU ownership are mocked in worker tests;
these are not fresh GPU measurements. The three-point fixture dispatches exactly
12 requests and retains all four-request/26-token and source/closure gates.

Commands from repository root (tests bounded externally to120s; review180s):

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none FERRULE_PROFILE_NVTX=0 /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/pressure_fault_v1 -p 'test_*.py' -v
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none FERRULE_PROFILE_NVTX=0 /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/tests -p 'test_execution_v2.py' -v
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none FERRULE_PROFILE_NVTX=0 /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/pressure_fault_v1/prepare.py check
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none FERRULE_PROFILE_NVTX=0 /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/pressure_fault_v1/subset_adapter.py check
```

Both CPU check commands passed: preparation retains12 unissued requests and
logical entry bounds `[8, 64, 128]`; candidate review retains9 prior middle-only
gates and requires all12 CLI runtime libraries. It reports zero GPU queries,
locks and runs. No execution command was launched.

The first test pass exposed a real post-observer expiry edge: HTTP terminal
success had been written before the context's exit deadline check. The private
serial seam now checks the phase clock before writing success evidence; the
regression requires censored plus three unissued slots. Shared code is unchanged.

## Seal and ownership

`plan_v2/plan.json`, `cases.json`, `requests.json`, `bindings.json`, `prepared.json`,
`cpu-probe.json` and `root-analysis.json` are separate preparation artifacts.
The cases and requests are exact checked copies for the same compiled86 candidate;
only the two existing coupled cache caps vary. The old root `plan.json` and
`bindings.json` are preserved and themselves pinned by the new seal.
`prepared.json` is an output containing the binding SHA, not a circular binding
input. All preparation files are made read-only after final verification.

`STAGE_FILES.txt` lists exactly19 owned source/docs/preparation files, including
unchanged historical root seals. Private raw artifacts and credentials are
excluded; no directory-wide staging. The new v2 entry refuses the historical
execution directory and permits only one new v2 GO. Source must be committed and
pushed before a future start. See `README.md` for the exact future commands;
**none of start/next/finalize has been invoked in this change**.

---

# Historical validation notes (unchanged record below)

# CPU validation — 2026-10-15

## Separately authorized command-entry extension

Passed **48/48 owned CPU tests** and **33/33 unchanged maintained execution
tests**, plus `prepare.py check` and `subset_adapter.py check`. The latter
rechecks the precise preserved build and all twelve required CLI libraries.
The initial extension test run caught the obsolete check-only CLI assertion;
it now asserts rejection of a general run loop and incomplete authorized
commands. No GPU was queried or launched by these checks.

The explicit parent GO permits adding the minimal `start` / `next` / `finalize`
entry before any pressure launch. `next` specializes the pinned maintained
finite-arm coordinator, not a new execution loop. The original CPU plan remains
non-authorizing. Fresh read-only private GO/cases/build-review/registration
receipts bind the separately approved scope, pushed source whitelist, exact CLI,
all eight UUIDs and the original 900/240/180s budgets. STOP on any error remains
mandatory, including a clean admission rejection. A second GO is disallowed.

Additional CPU tests cover denied GO/handoff, exact private coordinator
specialization with unchanged shared globals, expression drift, whole-point
reserve, exclusive read-only receipts, maintained authenticated supervisor entry,
and an actual maintained coordinator fixture that issues only the small four-slot
payload then stops on absent evidence. No fixture result is GPU evidence.

The precise preserved per-role build review is rechecked, including native
snapshots/toolchain/compiled sources; the CLI role must require all twelve runtime
libraries. Runtime maps still save birth-bound raw evidence before validation.
No historical receipt, production source, or shared protocol is changed. Private
raw execution is excluded from the ten-file source whitelist.

The earlier records below describe their original CPU-only passes; their old
ownership/entry status is historical, not the current command interface.

## Four-request budget adapter handoff

Status: **CPU test-prepared; GPU token blocked**. Parent accepted the limited
visible quality scope for coupled budget sensitivity. No fresh pressure or GPU
fault run, server, GPU query/lock, profiler capture, Git command, production edit,
shared-harness edit or historical receipt rewrite occurred in this adapter pass.

Passed **41/41 owned tests** (14 existing plan/evidence + 27 subset/worker tests)
and **33/33 unchanged maintained execution tests**. The worker integration uses
CPU fixtures only: subprocesses, HTTP transport, GPU telemetry and ownership are
mocked. It executes the actual maintained HTTP/serial/token-join/worker logic;
its synthetic observations are not candidate quality or pressure measurements.

Commands, from repository root (120s command bounds for test suites; 180s for
candidate review):

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none FERRULE_PROFILE_NVTX=0 /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/pressure_fault_v1 -p 'test_*.py' -v
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none FERRULE_PROFILE_NVTX=0 /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/tests -p 'test_execution_v2.py' -v
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none FERRULE_PROFILE_NVTX=0 /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/pressure_fault_v1/prepare.py check
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none FERRULE_PROFILE_NVTX=0 /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/pressure_fault_v1/subset_adapter.py check
```

The system `python3` initially failed to import `tomllib`; no workaround or new
dependency was added. Use the exact maintained base Python above. Integration
initially exposed the real 15s-idle-helper / frozen-10s-pre-idle incompatibility
and an empty-pin test fixture. The fixture now supplies real nonempty file pins.
The private idle adapter imposes the earlier absolute deadline without weakening
three-sample, foreign-memory/app, ECHILD or recorded-owner policy; shared helper
sources and the 900/240/180s budgets remain unchanged.

Coverage includes exactly four frozen slots, 26 completion tokens per lifecycle,
all three eight-ID canaries, actual HTTP/worker-ID joining, exact whole-log
admission/terminal order, no fifth tokenless admission, no extra token, no empty
PASS, wrong usage/IDs, partial/reordered/duplicate requests, censored vs unissued
journals, no retries, unknown custody, absolute deadlines, all-eight lock contract,
positive Ready/after-request map hooks, scoped source/caps before dispatch,
positive actual reader counts per point, cache charge bounds, preservation of the
shared 12-request validator, and restoration after failure. The three-point worker
fixture issues exactly12 requests; a wrong-point source report issues none.

Offline candidate review passed against the immutable compiled86 CLI
`7cc8e0ff6e84f27090494b8e9e27dc3a388d1d2f3fc34ae39dae416e1e607a01`,
its archive, prior raw nine-gate producers, precision/references, runtime pins and
current static cases. It does not call historical/current-HEAD source-cut
acceptance and does not recapture source or generation. **9/9 remains prior
middle64/320MiB numerical evidence only.** All new-point full-logit/route/state
quality is unverified; all visible pressure measurements remain not run.

Static formulas checked the existing actual-header payload3146112 and fixed
limits: metadata margin684032, double-buffer capacity56623104, requirement6292224;
logical positive entry bounds8/64/128. These are not newly executed CLI lowering
or CUDA admission at all caps. The source-options parser was also checked against
the preserved historical middle server log, without treating it as fresh pressure
evidence. Actual per-point source/owner admission remains pending server startup.

`bindings.json` seals the budget variant, exact candidate, owned/shared input
hashes and maintained interpreter/launcher/telemetry tools. `STAGE_FILES.txt` is
the exact ten-file whitelist; no blanket directory staging. Only the main agent
may stage/commit/push, after checking concurrent work. GPU ownership remains with
the profiling agent. A parent-authorized execution entry and its additional seal
must wait for GO and handoff; neither check command dispatches a worker. Fault
component selectors are unchanged and are not part of this pressure run.

## Earlier CPU preparation record (retained, not rerun here)

No GPU tests, server, profiler capture, model payload reads, GPU queries/locks,
Git commands or production/manuscript/other-harness edits were performed.
An existing CUDA-linked executable was invoked only with `--list`; both exact
selectors were listed without running tests. The CPU build used the repository's
existing ignored `target/` output, not production source edits.

### Earlier offline plan tests

Passed **14/14** before the subset adapter was added:

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none FERRULE_PROFILE_NVTX=0 python3 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/pressure_fault_v1 -p 'test_*.py' -v
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none FERRULE_PROFILE_NVTX=0 python3 -S -B papers/arxiv/experiments/storage_demand/pressure_fault_v1/prepare.py check
```

Checks cover preregistered denominators/order and fixed limits; canonical three
canaries; cache arithmetic vs admission; only two changed CLI flags; issued but
missing terminal = censored; missing metrics = null; known zero = zero;
quarantine is a subset of the charge; serialized snapshot copies have no live
owner/poll/join capability; worker's 12-count incompatibility; and source/receipt
hashes. `check` reports `cpu_plan_checked_execution_blocked`, no launch entry,
12 planned requests, logical bounds `[8, 64, 128]`, zero GPU queries/runs.

These tests check plan/evidence rules, not GPU performance or integrated resource
conservation. A serialized snapshot test does not stand in for actual owner
snapshot validation; the maintained real-worker tests below provide that narrow
component check.

## Maintained actual CPU worker — four exact selectors passed

First build/run, bounded to180s:

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none FERRULE_PROFILE_NVTX=0 FERRULE_CUDA_ARCH=sm_86 RUSTUP_TOOLCHAIN=nightly-2026-08-01 RUSTUP_AUTO_INSTALL=0 CARGO_BUILD_JOBS=2 cargo test --offline --locked -p ferrule-model --no-default-features --lib checkpoint::read_pipeline::tests::cancellation_after_start_does_not_release_running_read_or_pins -- --exact --nocapture --test-threads=1
```

One test passed; compilation took19.28s. Existing unrelated warning:
`PreparedNumericFp8RankLinear::payload_liveness` is unused. No fixes attempted.

Built executable:
`target/debug/build/ferrule-model/231d7ee33ed3978b/out/ferrule_model-231d7ee33ed3978b`

SHA256: `2d3f294c41adca71c696a943be6948644c9a7375e6ead0c7d36aaf9b3a1163c6`.
No-default-features CPU build; not the qualified serving CLI.

Three subsequent invocations used that executable, each with
`CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none FERRULE_PROFILE_NVTX=0`,
30s outer command bound, and `<exact selector> --exact --nocapture --test-threads=1`:

- `checkpoint::read_pipeline::tests::budget_snapshots_do_not_wait_for_or_refund_live_worker_stack`
- `transformer::expert_reads::tests::observation::receipt_identity_clock_and_all_charge_partitions_come_from_real_demand`
- `transformer::expert_reads::tests::observation::real_worker_notification_panic_finalizes_unknown_without_refund_or_healing`

Each passed1/1. The last intentionally panicked on the actual `checkpoint-read`
worker (`controlled real-worker notification failure`); the test then verified
retained packed/metadata charge, Unknown, and closed admission. It is not a CUDA
lost-consumer proof and does not share a full serving transaction owner.

Cancellation test: worker held at `IoStarted`, real pinned handle and eight-byte
reservation remain after cancel; completed source verification precedes payload
suppression; caller drop retires slot. Snapshot test:32 budget observations do
not wait/refund; stack stays charged until actual shutdown/join. Observation
checks do not alter reader counters or budget. These assertions are executed
component evidence, not an integrated serving E5 ledger.

## CUDA selector catalog — listed only

Existing executable:
`target/quality-lint-cuda124-sm86/debug/build/ferrule-model/9cb59307720193f3/out/ferrule_model-9cb59307720193f3`

SHA256: `762884263343f9df30b0895a2378ec25efd9532379a5587c1c25d8d77f89e180`.

For each of the two `minimum_gpu_selectors` in `plan.json`, the command used
`CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none FERRULE_PROFILE_NVTX=0
LD_LIBRARY_PATH=/home/admin/wxw403883/anaconda3/envs/tao/lib:/usr/lib64`, that
executable, and `--list --ignored --exact <selector>`;20s command bound. Each
reported `1 test, 0 benchmarks`. No test ran. This confirms names exist in a
local artifact, **not** that its full compiled source/native/runtime tuple
matches the qualified candidate. No GPU fault executable is admitted.

Future ignored tests need CUDA-supported hardware/runtime, explicitly bound
visible ordinal0, pinned same-source libtest/native/runtime artifacts,
`CUBLAS_WORKSPACE_CONFIG=:0:0`, `FERRULE_PROFILE_NVTX=0`, one test thread, and
coordinator-owned custody/deadlines. Explicit `FERRULE_CUDA_ARCH=sm_86` avoids
build auto-detection on the known target; it cannot authorize an unsupported
operation. Never substitute an SM103/FP4 provider for numeric-FP8 Mixed. Tiny
F32 `hybrid_unknown_finish` preserves the original freshly created core/topology
(frontier0); it does not demonstrate a previously successful nonempty committed
prefix under the 35B Mixed serving owner.

## Source and candidate limitations

`bindings.json` binds45 inputs, including CLI86 candidate final receipt,
mapfixed binding, corrected cases, actual-header metadata receipt, canonical
requests/oracles, relevant maintained adapter sources, and this plan/source/tests.
13 inspected model/composition source/test files match CLI86's archived source
pins. The later CLI source-admission test file does **not** match the archived
test file; that difference is explicitly recorded, not relabeled as same-cut.
The source-admission receipts are pre-existing metadata-only evidence, not newly
run admission at any pressure point.

Recorded candidate:9/9 gates passed; CLI SHA
`7cc8e0ff6e84f27090494b8e9e27dc3a388d1d2f3fc34ae39dae416e1e607a01`,
composition SHA `0a429c47c158bbe25f91730b75b47fd5cb768106f1d63d844da3a5f84fc8677f`.
Those binaries/configs are distinct from the CPU libtest and CUDA catalog
executable. Active `profile_overlap_v1` sources are excluded from the binding.
Pressure measurements, per-point effective CUDA admission, arbitrary-cap full
numerical quality, integrated Mixed serving faults and physical release remain
unverified. Parent decides narrowly scoped hooks/adapters; no autonomous GPU
follow-up or commits are authorized by these CPU results.
