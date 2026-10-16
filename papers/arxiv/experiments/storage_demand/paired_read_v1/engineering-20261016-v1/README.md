# Engineering closeout and same-cut upload experiment, v1

This is engineering evidence, not manuscript content. It includes completed GPU
quality stages and a partial actual same-cut whole-test comparison; the official
Nsight portion was blocked, so the24-job experiment did not pass.

## Completed actual GPU stages

| Stage | Immutable application commit | Actual tests |
|---|---|---:|
| Persistent read-worker lifecycle | `dae07b85c647cfc418b5412078874f9e8e9bf1db` | 9/9 |
| Backend public upload ticket API | `fc47c4173ca860b26f36774a3210f5a970a9e8b1` | 1/1 |
| Retained production upload integration | `57e46295d3cf5725605e076eb5543f6124bd9245` | 11/11 |

`completed-stages.json` contains bounded derived results, actual selectors/argv,
worker counts, numerical denominators, upload evidence and cleanup outcomes.
`source-pins.json` binds their frozen source archives, executables, controls,
authorizations/plans, logs, reports and supervisor/idle receipts by SHA-256.
Repository assets use repository-relative paths. The Python installation and
model mount are explicitly external environment dependencies, not public assets.

**The raw `target/validation` receipts are local ignored artifacts and are not
publicly distributed here.** Hashes do not make unavailable artifacts publicly
reproducible. No weights, executables, vendor tools, huge CSVs, private runtime
maps or authentication tokens are included. Derived summaries are not falsely
presented as byte-identical copies of their hashed original receipts.

All actual stages used exact-once selectors, zero retries/extra warmups, exact
UUID locks, authenticated single-thread subreaper/PID-birth custody, finite
bounds, ECHILD/drain and strict all-eight idle. CUDA was12.4.99/sm_86, not system13.
Quality retained ATOL0.00025/RTOL0.012 and exact routes/greedy/own-primary equality.
The constructor has no public post-drop witness: null/unobserved stays that way.
Live retained owners remain charged metadata8MiB+stack2MiB until a real final join;
process ECHILD is not an invented host-worker join or general CUDA release proof.

For both schedules, the retained Qwen stages demonstrated actual uploads:
2,089 full40,25,116 fresh,25,116 no-observer,11,354 cold-owner. Logical host peak
was36,701,896B under64MiB; sampled live idle charge10,485,760B. Cache current was
209,739,776B, peak219,754,496B under320MiB including scratch8MiB. Settled upload
and pending/Unknown charges were zero. These are NOT measured allocator/RSS/
memlock overhead, continuous alias/refund observations or a performance claim.

## Actual same-cut comparison:20 passed, official profiling blocked

`performance-results.json` records the actual approved57e4629 execution. Four
new policy/schedule diagnostic primaries and all16unprofiled trials passed.
Job21, the first official Nsight attempt, was blocked before model spawn; jobs22-24
were not dispatched. Command exit status was1 and the raw FINAL status is
`blocked`, not24/24. There were zero retries, warmups, calibration or rebuilds.

All8pairs showed a retained whole-test elapsed regression in this sample:

| Schedule | Pair | Synchronous seconds | Retained seconds | Retained/synchronous |
|---|---:|---:|---:|---:|
| blocking | 1 | 66.05 | 102.83 | 1.5569 |
| blocking | 2 | 65.96 | 101.37 | 1.5368 |
| blocking | 3 | 64.13 | 99.62 | 1.5534 |
| blocking | 4 | 66.18 | 100.31 | 1.5157 |
| lookahead | 1 | 61.61 | 99.91 | 1.6217 |
| lookahead | 2 | 65.68 | 99.21 | 1.5105 |
| lookahead | 3 | 61.96 | 97.58 | 1.5749 |
| lookahead | 4 | 62.86 | 97.74 | 1.5549 |

Median paired retained/synchronous ratios were1.5451 for Blocking and1.5649 for
Lookahead (54.51% and56.49% more whole-test time respectively). This is NOT a
speedup, a significance claim or token/TTFT/request latency. No default changed.

Full ordered64-prediction work arrays (one canonical array per schedule), the
64-call exact output-fingerprint map and per-stream check arrays are included
in the JSON. Each successful job binds their digests. All16own-primary and
10cross-policy comparison receipts matched, including8unprofiled pair checks;
all20successful reports had identical output fingerprints. Diagnostic numeric
routes/states and ordinary own-primary equality retained the approved criteria.

Both policies and schedules recorded25,116expert-cache upload operations per
test. Final live-owner read charge was10,485,760B (metadata8MiB plus stack2MiB). Host peak was
16,777,984B synchronous versus36,701,896B retained, both under64MiB; retained uses
explicit headroom16MiB/readiness30,000ms, not the default-off path. Cache final
current was209,739,776B and peak219,754,496B under320MiB, with64entries and8MiB
scratch. All132proof endpoints per ordinary trial and1,926per diagnostic primary
passed observed typed cache health, charge conservation, zero pending/Unknown and
zero outstanding retained upload charges. Live worker endpoints show1creation,2,559reuses and0joins;
a final-owner join is not invented from those still-live snapshots.

The profiler target failed `qualify.profile_auth` full-dictionary root identity
comparison: journal and authenticated root had the same PID253706, birth tick
145064963 and parent253705, but journal pgrp/session253705 versus live253706.
The maintained subreaper records its first child identity before synchronizing
with the child setsid; the preserved receipts/source establish this startup
snapshot mismatch. No control was changed or retried to bypass the rejection.
The root failed; one adopted owned tool descendant received authenticated
cleanup signals. All recorded children drained/reaped to ECHILD,45recorded
PID-birth identities were absent at closeout, all8GPUs were observed idle, and
all8local locks were released and reprobed. This was not a wholly natural exit
and is not a CUDA physical-release certificate.

**Official model sync/event/query/kernel/H2D/alloc/pin/unpin counts and costs are
unobserved, not zero.** The small preserved Nsight container is not accepted model
profiling evidence. There is no measured mechanism attribution for the regression.

## Approved separately reviewed24-run plan

Maintained entry: `../qualify.py`, promoted from the completed11 qualifier.
CPU contracts: `../performance.py`; `../retained_evidence.py` shares its settled
cache-health/type/charge checks with both policies and work-vector construction.
Retained-only policy, positive-upload and aggregate-minimum checks stay separate;
the synchronous arm does not inherit that minimum. `../quality.py` is unchanged.
The existing `../profile_audit.py` adds observed process-wide CUDA API summaries.
The existing `scripts/_profile_capture.py` adds an explicit natural-completion
mode and spawn-observation hook; its server default is unchanged. No second
Popen/reaping/kill engine or parallel benchmark harness was introduced.

Use the identical existing composition executable from57e4629 for both policies:
4new policy/schedule-matching diagnostic primaries,16unprofiled trials (4pairs
per schedule in fixed AB,BA,BA,AB order), then4separate official Nsight captures.
The Nsight version is2025.3.1.90. Each ordinary run binds its newly successful
same-policy/schedule/source primary by report hash. Ordered64-prediction work,
read/upload/cache and check arrays are recorded on both sides and must match.
Own-primary exact fingerprints remain mandatory; cross-policy bitwise equality
is descriptive, not a substitute or change to the fixed numerical criteria.

**Only whole-test elapsed time from the single-test libtest footer is measured.**
It includes construction, checkpoint IO, prediction work, checks and teardown.
It is NOT token, TTFT, ITL or per-request/model-only/serving latency. The printed
resolution is preserved. Enclosing supervisor wall time is recorded separately.
Four pairs per schedule do not support significance or p95/p99 claims. OS page
cache is uncontrolled; no cache drops, tuning, calibration, retries or outlier
exclusion are allowed. No default switch or six-fence speedup claim is made.

Profiler runs are excluded from all unprofiled paired summaries. A birth-bound
Nsight root owns a traced Python wrapper, which owns its unreaped model child.
The outer maintained subreaper alone owns descendant cleanup; no pidfd_open,
grandchild/group/foreign-PID shutdown is introduced. Actual model executable,
selector, ancestry, maps and SQL PID/UUID bindings are checked independently.

The approved profile was intended to capture observed sync/event-create/record/query/destroy/stream-wait
counts and host durations, query not-ready return counts (serial ready-spin
observations), kernel names/counts, H2D copy rows/bytes and alloc/pin/unpin API
overhead. CUDA API sums may overlap and are NOT elapsed latency. No per-prediction
NVTX ranges are invented: these immutable57 selectors do not initialize tracing.
An absent API/table never manufactures a zero-count or complete-trace assertion.

Exact source-declared absent-NVTX model, non-CUDA-wrapper no-payload and global
unsampled-scheduling diagnostics have a narrowly enumerated observed-count
scope. Every warning still prevents full-trace acceptance; unknown warnings,
CUDA absence on the model, event loss/errors and foreign GPU activity reject.
No full-trace completeness, CUDA overlap or exhaustive zero-sync claim is made.
These diagnostics and the owner/child topology require independent review GO.

## CPU checks and handoff

From repository root, with GPUs hidden and Python3.12 `-S -B`:

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/paired_read_v1 -p test_performance.py -v
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s scripts/tests -p test_profile_capture.py -v
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s scripts/tests -p test_bench_profiling.py -v
```

Preparation performs CPU pin/package checks only; it never rebuilds releases.
The ignored stage is `target/validation/retained-performance-57e4629-20261016-01`.
Its `review-ready.json` binds the exact control delta, plan, CPU receipts and
runtime/tool pins. A missing or mismatched independent review record blocks
before any GPU locks, telemetry or dispatch. The actual authorized execution
above stopped at job21; do not resume or rerun this exact-once stage.
`validation.json` is the pre-execution CPU review receipt, including its then-zero
GPU count. `performance-results.json` is the actual execution/partial-result record.
The ignored `actual-closeout-01/approved-curated-snapshot/` preserves exact approved
pre-publication metadata; the approval packet itself was not rewritten.

The main agent handles staging/commit/push. `/papers/` ignores new files globally,
so the explicitly listed new maintained controls and this bounded stage directory
need selective `git add -f`; do not add raw target/vendor outputs or alter ignore
rules. This executing agent has not staged, committed or pushed anything.

Exact refreshed main staging hashes are in the local ignored
`target/validation/retained-performance-57e4629-20261016-01/actual-closeout-01/stage-files.json`.
Only README.md, performance-results.json and manifest.json changed for this
actual-results closeout; executable controls remain byte-identical to approval.
