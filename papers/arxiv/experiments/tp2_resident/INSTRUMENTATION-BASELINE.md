# Collector-first instrumentation scope (CPU-tested executable wiring)

No collector, model, GPU query, UUID lock, Nsight probe or production change was
performed for this update. Another runtime owner is using the GPUs. This is not
an extension of the previous grant and makes no assertion about current GPU idle.

## What the installed Nsight documentation actually says

The saved 2026.1.3 `profile --help` and installed `UserGuide/index.html` support:

- `--gpu-metrics-devices=none` (already the default)
- `--gpu-video-devices=none` (already the default)
- `--gpuctxsw=false` (already the default)
- `--cuda-trace-scope=process-tree`

The maintained profiling command now declares those settings explicitly. This
is **not evidence that NVML/CUDA enumeration is disabled**. No documented
no-enumeration switch was found. Device metrics `none`, CUDA visibility, and
application-scope traces do not imply the profiler never discovers other cards.
The previous 2/4 MiB nonselected deltas therefore remain unexplained observations,
not proof that all-card metrics were enabled or that Ferrule used eight GPUs.

The installed guide documents named interactive `nsys start --session-new=...`
then `nsys launch --session=...`, and launch-before-start workflows. Thus a
collector-first architecture is possible in principle. Whether device discovery
finishes before the selected model or trace window starts must be measured;
CLI docs and the old CPU-only sleep probe do not prove that condition.

## Bounded probe, with a distinct eight-GPU grant

`instrumentation_scope.py` implements CPU-testable policy/lock helpers.
`instrumentation_live.py` binds private hooks in `paired.py`; `run`, `worker`,
and `requests` remain the existing `experiment_current.py` implementation.
`instrumentation_capture.py` is a gated non-CUDA target that imports the existing
`_profile_capture.run_target`, not a copied server process/signal loop. These
files are **not a launch authorization**. It requires a new grant named
`all_eight_instrumentation_probe`, with all eight actual physical UUIDs and an
operator note. An existing two-GPU grant cannot substitute. UUID discovery must
be reverified live under the next authorized owner; no UUIDs are guessed here.

1. Acquire all eight UUID locks in sorted order using the existing
   `ct.exclusive_owner_lock` through `ExitStack`; any acquisition failure releases
   already acquired locks. Never signal the holder. Retain all eight through
   final collector drain; the model visibility stays the original ordered two.
2. Under the existing authenticated subreaper, record the original all-eight
   zero-memory/no-compute-PID/zero-utilization baseline **B0** before collector
   initialization. Lock cooperation is an operational exclusivity assumption,
   not proof against a noncooperating foreign process or host-NVML PID mapping.
3. Start only the collector plus a gated non-CUDA target; do not start/import
   the model yet. Bind root PID/start-time/launch ID and retain live descendant
   authentication. No detached global session may escape that custody.
4. Within **60 s**, require at least **three identical per-card memory samples
   spanning >=4 s**, complete UUID/index mapping, no compute PIDs (including
   unknown/unmapped PIDs) and zero utilization. Predeclare an instrumentation-only
   ceiling of **16 MiB/card**. This is a proposed bounded probe allowance, **not**
   a measured value or a replacement for the model's 1 MiB delta tolerance.
   Instability, missing samples, new PIDs or ceiling overflow blocks model launch.
5. Freeze **B1** exactly once, recording B0, B1 and B1-B0 separately. Recheck live
   locks/custody, then open the model gate only with selected UUID0/1 visibility.
   Do not choose a maximum/median to hide instability, import an earlier run's
   B1, reacquire between phases, or change B1 after model start.
6. Model-phase nonselected positive deltas remain **<=1 MiB relative to B1**;
   nonselected utilization or compute PIDs stop the owned run without foreign
   signals. Selected cards retain the **absolute 22 GiB/card** ceiling; do not
   subtract profiler allocation to make the model fit. Host accounting includes
   the collector tree and remains observational, not a physical partition.
7. Ready/Hello/canaries precede capture. Trace one warm QA8 and one diagnostic
   measured QA8; stop capture and export. Startup stays outside request traces.
   If model launch or trace start causes another nonselected allocation spike,
   **fail, preserve it, no adaptive rebaseline**.
8. After authenticated model drain, prove return to the frozen collector anchor.
   Drain the collector through the same custodian, prove return to original B0
   on all eight, and only then finish release. On failure, cleanup remains the
   authenticated owner/custodian's responsibility; unknown drain is not success.

The tests cover separate grant, sorted locks/partial rollback, stable/unstable
baseline, unknown PID, deadline, ceiling, immutable baseline, two-card model
visibility, root identity, unchanged 1 MiB delta rule and original-baseline return.
`BaselineScope` validates metadata/state transitions only; the sealed worker must
supply actual live authentication and drain receipts, never caller pass flags.

### Implemented handoff and remaining live validation

The profiling-purpose adapter now requires `--probe-grant` in addition to the
usual renewed sole-owner grant. It verifies that grant before any query/lock,
acquires eight sorted existing UUID locks, records `probe-locks.json`, and runs
the same sealed worker/subreaper. Ordinary smoke/pair retains exactly two locks.

Before Nsight launch the sealed worker writes `collector-manifest.json`, binding
its existing authenticated launch, coordinator's live lock FDs, B0, selected
command/hash and nonce. Nsight launches the non-CUDA target, which must be a live
kernel-identified descendant of that worker. It publishes target PID/start-time;
it cannot launch the model until the worker samples B1 and publishes a read-only
`model-release.json` by atomic exclusive link. Both sides recheck the live
custody journal, coordinator/helper/worker identities, all eight locks, target
ancestry and manifest/nonce/launch binding. The gate freezes B1 once. Samples and
failures go to `collector-samples.jsonl`, not discarded on exceptions.

After the original model wrapper reaps its direct child, the gated target stays
alive and publishes `collector-model-ended.json`. The worker shutdown hook
samples B1 because the original lifecycle stops its main monitor before shutdown.
It records `collector-after-model.json` only after two exact anchor-return samples,
then releases the target and waits for Nsight export. The original subreaper
still proves descendant drain. Final idle requires original **all-eight zero B0**;
`probe-lock-release.json` separately records eight-lock closure, drain and B0
status (the original two-model-UUID release receipt remains unchanged).
Failure permits owned cleanup, not a successful B1/B0 claim or foreign signals.

Startup900 includes the <=60-second collector gate, requests120/all-requests440,
teardown120, supervisor1600 and overall1700 remain bounded per lifecycle. There
is no stability retry, transport fallback, model restart or adaptive rebaseline.
Trace start/stop are the documented named-session operations; model launch or
trace-start allocations above B1+1 MiB on any nonselected device still abort.

**Unverified until renewed GPU authorization:** whether this installed collector
fully initializes before model/trace start, exact return to B1/B0, and CUDA
coverage for both TP2 owners. CPU tests do not establish those facts and do not
promise a successful trace. The 16 MiB instrumentation ceiling is predeclared,
not a measured or silently expanded model tolerance.

## Observer memory decomposition and phases

The existing observer now parses `Anonymous`, `Pss_Anon`, `Pss_File`, and
`Private_Dirty` from `smaps_rollup` in addition to old RSS/PSS/locked/pinned fields.
Optional fields absent on older kernels are **null**, not zero; aggregate PSS
coverage remains valid when an optional breakdown is unavailable. Per-component
process/sample coverage is retained. A vanished descendant still makes whole-tree
memory incomplete, not root-custody loss. `status` cannot override rollup values.

`Anonymous` is resident anonymous memory, not proportional anonymous PSS;
`Pss_Anon + Pss_File` may omit shared-memory PSS. Independent component peaks do
not add up to the total PSS peak. Do not reinterpret file-backed PSS as anonymous
weight duplication or infer residency/offload solely from these counters.

Each sample records its start and end. The maintained paired/smoke/profile adapter
marks readiness using the existing readiness helper. `hello2/http.json` supplies
actual first HTTP send/EOF timestamps. New additive `phase_memory` separates:

- startup (before Ready)
- Ready-to-first-request
- first request
- after first successful request

Only wholly contained samples belong to a phase; straddling/missing timestamps
stay unassigned/raw. Empty windows and unsupported component coverage are null.
`after_first_request` includes later canaries/warmups, so it is an explicit
**post-first observational window, not proof of steady state**. Original lifetime
peak keys remain unchanged, and old receipt files are not rewritten. Older
callers without a Ready marker retain lifetime/post-first evidence but no invented
startup peak. The old consumed adapters are archived under
`history/pre-instrumentation-baseline/` before these edits.

## Optimized release follow-up

The ongoing runtime optimization's binary/source/quality results are separate.
A new optimized release still needs matching source/build/numerical admission and
same-release serving smoke, then a **fresh no-trace paired pilot**. Do not borrow
the old release's serving receipt or use profiler durations as benchmark results.
E2 remains incomplete; `fairmatch=false`, `exact_speedup=null`.

New executable CPU plan:
[collector-handoff plan](plans/collector-handoff-20261009/plan.json).
The [earlier policy-only plan](plans/instrumentation-baseline-20261009/plan.json)
remains historical documentation, not current integration status.

## Executable entry points

**These commands are documented, not GPU-authorized or executed in this update.**
CPU-hidden `optimized_release.py bootstrap` has already produced
`admission/optimized-20261009/smoke-admission.json` and a read-only independent
staged copy of release SHA
`4d40c34738d36a1c78f62c0e400dd732da40e34c316ce4bd3c356dc4d29b3173`.
The audit checks the actual optimized builder schema: 19 successful gates,
37 calls / 40,293 exact metrics, successful authenticated test drain/release,
production/source relation, binary and dynamic library pins. It preserves the
one failed GPU gate attempt. It does not run comparators or regenerate HF.
Canonical manifest/comparator hashes are checked; all numerical tensor payloads
are **not recomputed/rehashed by this serving-admission audit**. The builder's
raw Cargo output is terminal-only, not an invented on-disk build log or
an authenticated build-supervisor receipt.

`runs/tp2-optimized-smoke-20261009-001` is **already CPU-prepared and validated**,
not launched. Do not repeat its preparation command below; it is retained as the
reproducible creation recipe. The next authorized step for that ID is `run`.

Use the same pinned base Python (`-S -B`, GPU-hidden parent) for every command:

```sh
# CPU-only fresh smoke preparation (do not repeat an already prepared ID).
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none \
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
papers/arxiv/experiments/tp2_resident/paired.py prepare --purpose serving_smoke \
  --id tp2-optimized-smoke-20261009-001 \
  --admission papers/arxiv/experiments/tp2_resident/admission/optimized-20261009/smoke-admission.json
```

Only after renewed two-GPU sole-owner authorization: invoke the same `paired.py`
with `run --purpose serving_smoke --id tp2-optimized-smoke-20261009-001
--sole-gpu-owner-granted --owner-note '<actual authorization>'`. This runs S1/S2
through the sealed supervisor: Hello2, three frozen max8 canaries, cross-request
reuse, cancellation/post-cancel reuse, normal exit and a rebuilt CPU owner.
No same-release HTTP receipt is required to *start* this smoke.

Only after `serving-smoke.json` actually passes:

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none \
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
papers/arxiv/experiments/tp2_resident/optimized_release.py promote \
  --smoke-admission papers/arxiv/experiments/tp2_resident/admission/optimized-20261009/smoke-admission.json \
  --smoke-receipt papers/arxiv/experiments/tp2_resident/runs/tp2-optimized-smoke-20261009-001/serving-smoke.json
```

This CPU-only command verifies same-binary/source/spec HTTP evidence, reuses the
pinned vLLM baseline admission, and exclusively creates `paired-admission.json`
and `profile-admission.json` in that admission directory. Missing/old-release
smoke fails closed; no placeholder receipt is generated.

Prepare fresh `paired.py --purpose paired` with `paired-admission.json`, then
explicitly authorize its `run`. This is **no-trace** F1,V1 / V2,F2, each Hello2 /
three canaries / excluded first QA8 +2 warmups +5 measured. Same FP8 source,
BF16 boundaries/F32 SSM, TP2/PP1/EP1, context128/batch8/seq2/nooffload; same UUIDs.
Record CPU collective vs NCCL, Mixed/HF reference vs Marlin, and effective Ferrule
decodebatch1. No correctness/profiling wall time enters performance results.
After completion, `compare_pilots.py --new <fresh-paired-run-directory>` creates
an offline `old-vs-optimized.json` with per-slot startup/actual tokens/TTFT/E2E,
GPU/PSS and phase smaps (old missing fields null), no exact speedup or statistics.

For a separately authorized diagnostic profile, prepare fresh
`paired.py --purpose profiling --admission .../profile-admission.json --id ID`.
The owner then supplies a new JSON grant whose fields are:

- `scope`: `all_eight_instrumentation_probe`
- `granted`: `true`, with a nonempty `note` stating the actual new authorization
- `run_id`: that exact prepared ID
- `prepared_sha256`: SHA256 of that run's `prepared.json`
- `gpu_uuids`: **sorted eight actual physical UUIDs** (no fabricated inventory)
- `model_gpu_uuids`: ordered
  `GPU-b980f22b-58a3-fc88-21c1-ae86053d8535`,
  `GPU-63f1bd01-ddf9-a742-ee48-9389892493d9`

Run via the existing entry:

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none \
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
papers/arxiv/experiments/tp2_resident/paired.py run --purpose profiling \
  --id <prepared-ID> --sole-gpu-owner-granted --owner-note '<new authorization>' \
  --probe-grant <run-bound-eight-UUID-grant.json>
```

The grant is checked **before acquisition/query**, copied read-only into the run,
and bound into the sealed target manifest. B0 under all eight locks rechecks the
actual UUID inventory and selected physical indices. The eight locks authorize
instrumentation observation only; model environment remains selected2. Unknown
NVML PID mapping is still unknown, not owned-byte attribution. No foreign signal
is introduced by any new hook. Old M7/G10, canary-failure and failed-profile
receipts remain untouched; pre-edit maintained files were archived under
`history/pre-collector-handoff/`.
