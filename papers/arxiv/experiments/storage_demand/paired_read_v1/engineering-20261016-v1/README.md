# Engineering closeout and same-cut upload experiment, v1

This is engineering evidence, not manuscript content. It includes completed GPU
quality stages, the negative same-cut whole-test comparison, preserved blocked
profiling attempts, and four later separately approved official Nsight captures.
Those four pass execution/quality/custody gates only: their observed-model
summaries remain PARTIAL, not accepted full traces. The original24-job stage
remains blocked and was never resumed or relabeled successful.

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

## Historical same-cut comparison:20 passed, first profiling attempt blocked

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

**That original attempt has no qualified model API counts.** Its small preserved
container was not accepted model profiling evidence. It provides no mechanism
attribution. The later failed post-setsid capture and its offline diagnostic-only
export are preserved in `performance-results.json#/profile_failure_log`; neither
historical failure was promoted by the separately approved captures below.

## Actual four-profile closeout: partial observed-model evidence

`profile-results.json` records four fresh captures from the same immutable57e4629
composition executable, using official Nsight2025.3.1.90. The separately approved
stage was `target/validation/retained-profile-57e4629-natural-drain-20261016-01`;
all4 gates passed exactly once with no retries, warmups, calibration or builds.
Each compared9,187,840 common-history logit values, maintained its exact own-primary
fingerprints/greedy checks, and matched the canonical ordered64-prediction work,
read/upload and per-stream check arrays already published in `performance-results.json`.
Each policy executed25,116 logical uploads. These are real opt-in retained runs,
not results from the unchanged default-off branch.

The post-setsid root handshake, bounded natural descendant grace and final
terminal WNOWAIT check were independently reviewed. All4 captures exited naturally
without signals: each custodian reaped4 authenticated identities, then ECHILD.
Adopted descendants drained in about0.31-0.64s within the existing5s bound and original
job deadline. All8 GPUs were observed idle before/after; final ECHILD/owner absence
and all8 lock releases/reprobes passed. No CUDA physical-release certificate is
inferred. The two prior consumed stages and all8 negative timing pairs remain unchanged.

For each policy, the API counts listed below matched across schedules. Query-spin and context-selection counts differed. All counts are whole-model-process observations. Synchronous versus retained:

| Observed API | Synchronous calls | Retained calls | Delta /25,116 logical uploads |
|---|---:|---:|---:|
| Host-pin allocate | 2 | 25,118 | +1 |
| Host-pin free | 2 | 25,118 | +1 |
| Event create / record / destroy, each | 347,563 | 372,679 | +1 each |
| Stream wait-event | 1,387 | 26,503 | +1 |
| Stream synchronize | 382,743 | 232,047 | -6 |
| Event synchronize | 77,904 | 77,904 | 0 |
| Context synchronize | 3 | 3 | 0 |
| Device allocate / free, each | 34 | 34 | 0 |
| H2D copies | 195,985 | 195,985 | 0 |

| Schedule | Pin+free host sum, sync | Pin+free host sum, retained | Added pin+free host sum / logical upload | Stream-sync host sum, sync → retained |
|---|---:|---:|---:|---:|
| Blocking | 0.003209s | 29.853418s | 1.188494ms | 4.340770s → 0.302237s |
| Lookahead | 0.003164s | 29.472179s | 1.173316ms | 4.344680s → 0.302241s |

H2D API host sums decreased5.512→1.150s Blocking and5.523→1.134s Lookahead.
H2D device counts/bytes were unchanged:195,985 copies /82,512,273,944B; summed
copy device durations were3.854→3.965s Blocking and3.850→3.968s Lookahead.
Kernel names/counts matched (409,411); summed actual kernel intervals were
3.378→3.399s Blocking and3.380→3.399s Lookahead. These interval sums are not
whole-test elapsed time, and copy/kernel time need not be additive.

Ready-query NOT_READY600 counts increased18,210→69,553 Blocking and
20,104→70,451 Lookahead (added2.044 /2.005 per logical upload). Total query host
sums were0.158→0.210s and0.155→0.209s respectively. Retained model-process OSRT
also recorded50,129 /50,167 `clock_nanosleep` calls, about5.213 /5.216s summed
wall intervals (~2 observed calls per logical upload). Synchronous exports had
no matching sleep rows: that remains unobserved, not a semantic zero. Without
prediction/upload ranges or call stacks, these sleeps are not proven to be
upload-pending sleeps. CPU initial zero/packing costs and completion gaps remain
unobserved. Other OSRT calls and per-return query durations are in the JSON.

Live proof endpoints retained metadata8MiB+stack2MiB charge10,485,760B. Host peak
was16,777,984B synchronous /36,701,896B retained under64MiB. Cache current/peak
209,739,776B /219,754,496B stayed under320MiB including8MiB scratch. All132
sampled endpoints per run passed settled pending/Unknown/upload-charge checks.
The committed serializer omits upload-charge fields only when zero; this is not
a missing-observation waiver. Each live endpoint showed1 worker creation,
2,559 reuses,0 joins and an alive worker. A post-owner-drop join witness is not
exposed here and remains null/unobserved; neither ECHILD nor still-live zero joins
is substituted for that witness.

**Scope:** all summaries are `PARTIAL_OBSERVED_MODEL_SUMMARY`, full-trace acceptance
false, dropped events null and exhaustive API counts unproven. All248 warnings per
capture are retained/classified by exact Analysis/HostTimestamp mechanism, subject,
SQL binding and effect. Unbound subjects remain not owned. Scheduling-absent means
no CPU scheduling claims; no NVTX means no expert/range attribution. Ordinary NULL
copy endpoints remain unknown. H2D size buckets do not uniquely certify the six
expert parts. Raw ignored containers/SQLite/full diagnostics remain local-only;
the compact JSON publishes their hashes, bounded observations and canonical-array
references, not unavailable raw evidence as a public reproducibility claim.

**Engineering conclusion:** per-upload pin/free churn is a directly observed
large host API outlay consistent with the measured regression. The partial
profile sums overlap across threads/OSRT and are not an exclusive causal account
of the earlier unprofiled elapsed differences. The prior1.5451× /1.5649× negative
paired medians remain the performance result; profiled footer times do not enter
that population. Keep the default unchanged. Prioritize separately reviewed
bounded pinned-staging reuse with full quota/lease/consumer-proof accounting and
then quality plus measured A/B; grouped-event work alone is not a speedup claim.

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

Historical closeout staging hashes are in the local ignored
`target/validation/retained-performance-57e4629-20261016-01/actual-closeout-01/stage-files.json`.
Only README.md, performance-results.json and manifest.json changed for this
actual-results closeout; executable controls remain byte-identical to approval.

For this later profile closeout, only `profile-results.json`, `README.md` and
`manifest.json` change. Current selective main staging hashes are in
`target/validation/retained-profile-57e4629-natural-drain-20261016-01/actual-closeout-01/stage-files.json`.
All executable controls remain byte-identical to the four-capture approval.


## Grouped synchronous upload quality (`ce6d4ff`)

`grouped-quality-results.json` records the separately reviewed, actual14 GPU gates
on immutable `ce6d4ff28b96d4057d0de6dfa858707104aad16b`: backend1, model5 and
composition8 all passed once, without retries, extra warmups or numeric/quota
relaxation. The exact selectors, command, cut/artifact/control receipts and raw
ignored-target references are recorded there. Raw target output is not public.

Both schedules passed full40, fresh generation, exact no-observer fingerprints
and cold-owner restart against the unchanged references and ATOL0.00025/RTOL0.012.
Fresh and no-observer runs each observed25,116 cache uploads, one persistent
worker creation and2,559 reuses. Their logical host peak was16,777,984B under the
same64MiB cap; settled live-owner charge remained10MiB (metadata8MiB +stack2MiB).
Cache current/peak maxima were209,739,776/219,754,496B under320MiB including scratch.
Upload charges, pending cache uploads and Unknown bytes were zero at proof
endpoints. No additional retained pin/ticket reservation is used by grouped mode.

The passing backend real-GPU contract asserts6 copies and1 synchronization per
grouped call for both scale types. Model-wide group synchronization counts and
post-final-owner join witnesses are unobserved, not invented. All14 jobs exited
naturally with authenticated ECHILD and owner absence; strict all8 idle and final
lock release/reprobe passed. These do not certify physical CUDA release.

This is quality evidence only: default remains unchanged, no performance or
overlap claim. The prior57e4629 negative timing pairs and partial official
profiles remain unchanged and must not enter a new ce6d4ff performance population.
