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


## Grouped whole-test comparison: completed pairs, failed profile gate (`ce6d4ff`)

`grouped-performance-results.json` closes the separately approved fixed24 plan
from `target/validation/grouped-performance-ce6d4ff-20261017-01/`. **The stage
failed, not24passed:** jobs1–20 passed (four diagnostic primaries and16 unprofiled
trials); profile21 failed and22–24 were not dispatched. There were no retries,
extra warmups or calibration. This offline closeout issued no GPU jobs, builds,
new captures, production/control changes, commits or pushes. The original
negative eight57e4629 retained pairs, historical partial profiles and actual14
grouped-quality evidence remain byte-identical and outside this timing population.

The executable remains frozen `ce6d4ff28b96d4057d0de6dfa858707104aad16b`,
SHA256 `d8c0bae4872ffbbb6828dd7f49acecb4de0f69ab32748662845ddaa6d0b30557`;
the archive SHA256 is `4e15b7301f678424878ba97c3a3df8f8533a696053d2eb58b36bb35e6895e135`.
The closeout worktree HEAD is not the application cut. The JSON includes frozen
source-file pins, all trial/primary times, ordered64 work/check/fingerprint arrays,
case IDs, per-report hashes, effective configs and27 equality receipts (including
the failed profile's numerical-only comparison). All20 successful reports pass
the unchanged ATOL0.00025/RTOL0.012 checks. All21 numerical reports share exact64
output fingerprints; each schedule's ordered work/check arrays match across arms,
repetitions and its primary. Lookahead speculation differs from Blocking by design.

### Descriptive completed unprofiled pairs only

| Schedule | Pair | Synchronous whole test (s) | Grouped whole test (s) | Grouped / synchronous |
|---|---:|---:|---:|---:|
| Blocking | 1 | 66.92 | 63.93 | 0.955320 |
| Blocking | 2 | 64.61 | 62.50 | 0.967343 |
| Blocking | 3 | 65.17 | 61.84 | 0.948903 |
| Blocking | 4 | 66.51 | 65.55 | 0.985566 |
| Lookahead | 1 | 62.10 | 60.05 | 0.966989 |
| Lookahead | 2 | 63.62 | 60.77 | 0.955203 |
| Lookahead | 3 | 63.19 | 57.61 | 0.911695 |
| Lookahead | 4 | 61.25 | 59.53 | 0.971918 |

Median within-pair ratios are **0.961331 Blocking** and **0.961096 Lookahead**.
Grouped has lower elapsed time in all eight observed pairs, but these are only
four pairs per schedule, printed at0.01s resolution, including IO, construction,
numerical checking and teardown. They are **not TTFT, token or request latency**,
not significance/quantile/causal evidence, and do not enable a default. Primaries
and profiles are excluded. The maintained exact24 accepted-summary guard still
rejects this failed stage; the table is explicitly an offline descriptive summary
of the completed pairs, not an accepted full-stage result.

Each numerical report observes25,116 logical cache uploads,25,116 verified reads,
79,017,748,992 proved read bytes and150,696 reader intervals. Logical host peak
is16,777,984B under64MiB; final live-owner charge is10MiB (metadata8MiB +stack2MiB),
with one worker creation and2,559 reuses. Cache current/peak maxima are
209,739,776/219,754,496B under320MiB including8MiB scratch. Upload/pending/Unknown
charges are zero at validated proof endpoints. Grouped adds no retained pin/ticket
reservation. These are logical charges, not RSS or physical CUDA-release proof;
post-final-owner join and model-wide grouped synchronization counts are unobserved.

### Profile21 remains failed; existing offline export is diagnostic only

The recorded error is `ValueError: Sticky profile identity/observer error`.
The observer log contains36 identity-bound model samples, then
`CustodyLost: Authenticated server root disappeared`. Three early mapping
assessments lacked required `libcupti.so.12.9`;33 later assessments passed.
The wrapper records `natural_completion` and application return0; the outer
supervisor records return0, natural drain without signals, authenticated ECHILD,
owner absence and strict all8 idle. **None of this waives the sticky failed gate.**
No profiling fix, retry, restart or jobs22–24 were attempted in this closeout.

The prior agent already exported the failed container to
`actual-closeout-01/failed-capture.sqlite` and audited it CPU-only; takeover
verified and reused that output without another export. The JSON retains counts,
API host-duration sums, kernels and copy rows under
`INCOMPLETE_FAILED_CAPTURE_DIAGNOSTIC_ONLY`, never as an accepted paired profile.
For example, the failed synchronous trace contains382,743 `cuStreamSynchronize`
rows and2 each host-pin allocation/free rows. No grouped comparator exists, so
**accepted profile metrics and arm deltas are unobserved, not zero**. The proposed
−5 synchronizations/upload remains a hypothesis; no expert-range, API-exhaustiveness,
dropped-events=0, full-trace, overlap or causal attribution is justified.

Historical final cleanup receipts show three consecutive all8-idle samples,
authenticated ECHILD, owner absence and eight local lock releases/reprobes. Offline
PID/birth inspection found no surviving owned task; no new GPU query or foreign
signal was issued. This does not certify physical CUDA release or global reservation.

**Availability:** raw ignored `target/validation/` reports, receipts, source archives,
binaries, profile container and SQLite remain local-only; hashes do not imply
public availability. Only the curated derived result, this README and manifest
are closeout staging candidates. No private weights, vendor packages, raw target
files or binaries are included. `grouped-performance-results.json` is ignored and
requires selective force-add by the reviewing main agent, not recursive target staging.
CPU closeout validation uses the existing80 control tests and full frozen-pin/GO
checks with the pinned Python3.12, plus manifest and raw-receipt/hash checks.
The default synchronous path and prior engineering conclusions remain unchanged.


## Fresh grouped profiles: three qualified captures, fourth failed

`grouped-profile-results.json` records the actual, separately approved four-profile
execution in `target/validation/grouped-profile-ce6d4ff-terminal-handoff-20261017-01/`.
**The stage failed:** Blocking synchronous21, Blocking grouped22 and Lookahead
grouped23 passed; Lookahead synchronous24 failed. All four were issued once;
no retries, warmups, release builds, original20 timing/primary reruns or quality14
reruns. No further GPU work followed. Frozen ce6d4ff application/archive/binary
are unchanged; committed control d48cad78 uses the reviewed258b12c terminal fix.

The three passing captures validate exact job/plan/launch/model/wrapper/ancestry
terminal receipts, stopped observer acknowledgements, final owner wait ordering,
signal-free natural custody, authenticated ECHILD and all8 idle. Their observer
acks followed terminal observation by1.395/1.439/1.427s, within15s and the original
capture deadlines. The pinned wrapper performs its single authoritative
`Popen.wait` only after the successful ack; no independent runtime wait counter
is claimed. Terminal/ack receipt hashes and ordering are retained per capture.

### Qualified Blocking pair: whole-model-process observations

Both reports record25,116 logical uploads and identical ordered work/check arrays
and64 exact output fingerprints against their own completed primaries.

| Observation | Synchronous21 | Grouped22 | Grouped − synchronous / upload |
|---|---:|---:|---:|
| `cuStreamSynchronize` rows | 382,743 | 257,163 | −5 |
| `cudaStreamSynchronize_v3020` rows | 290 | 290 | 0 |
| Host-pin allocation rows | 2 | 2 | 0 |
| Host-pin free rows | 2 | 2 | 0 |
| H2D copy rows | 195,985 | 195,985 | 0 |
| H2D bytes | 82,512,273,944 | 82,512,273,944 | 0 |
| Observed kernel rows | 409,411 | 409,411 | 0 |

The stream-sync reduction is125,580 rows, exactly5 per logical upload, matching
the predeclared hypothesis **without using it as an acceptance threshold**.
Both pin/free counts match the expected2 per model process; all69 observed
kernel-name/count entries match exactly. Event-query and context-selection
counts differ; the JSON retains every observed API's count, host-duration sum,
normalization and Blocking delta, plus copy bytes and kernel-map hashes.

`cuStreamSynchronize` host-duration sums are4.343492514s versus1.597095312s
(−109,348.511ns / logical upload). These are process-wide host sums that may
overlap other work, **not elapsed savings or per-expert/per-upload latency**.
No expert NVTX/range attribution, causal or default-enabling claim follows.

Lookahead grouped23 is qualified but unpaired:257,163 stream-sync rows,2 each
pin/free rows, and the same observed H2D/kernel work. **Qualified Lookahead arm
deltas remain unobserved, not zero.** The prior eight unprofiled whole-test pairs
and their approximately3.9% descriptive median difference remain byte-identical
in `grouped-performance-results.json`; none of these profiled times joins that
population, and no TTFT/token/request latency or significance is inferred.

### Failed24: sticky observer error is not waived

The model's WNOWAIT receipt records exit0, but an earlier observer sample saved
`FileNotFoundError: [Errno 2] No such file or directory: '/proc/56467/exe'`.
Its ack has status `blocked` and retains that sticky error. The wrapper failed
with `Missing/blocked/foreign profile observer acknowledgement`; successful
`profile-model-ended.json` is absent. Supervisor return1 and owned failure-cleanup
signals mean **this was not signal-free natural completion**. No source fix,
retry or recollection was attempted. This evidence does not establish a new
root-cause diagnosis and does not rehabilitate any failed historical capture.

The existing failed container was exported once CPU-only to ignored
`actual-closeout-01/failed-capture.sqlite` and read-only audited. Its382,743
stream-sync rows,2 each pin/free rows and matching H2D/kernel observations are
retained solely under `FAILED_CAPTURE_DIAGNOSTIC_ONLY_NOT_QUALIFIED`, with no
accepted Lookahead delta. SQL completeness does not waive the blocked ack.
The original fixed24 profile21 failure is also retained unchanged.

All four numerical reports pass the original tolerances/work/fingerprints and
observe25,116 uploads each. Host peak16,777,984B remains below64MiB; live-owner
settled charge10MiB, cache current/peak209,739,776/219,754,496B below320MiB, and
validated endpoint upload/pending/Unknown charges remain zero. Numerical report
success does not override failed profile custody.

Final authenticated ECHILD, owner absence, three consecutive all8-idle samples
and all8 local lock release/reprobe passed. Offline PID/birth checks found no
surviving owned tasks. The failed fourth required maintained owned cleanup; no
manual or foreign signal was issued. Physical CUDA release remains unobserved.

All evidence is **PARTIAL_OBSERVED_MODEL_SUMMARY**: full_trace_accepted=false,
dropped_event_count=null, API exhaustiveness unproven; no CPU scheduling, expert
ranges or overlap claims. Raw ignored receipts/traces/SQLite/binaries remain
local-only, not public; no private weights or raw target files are staged. Only
this README, `grouped-profile-results.json` and `manifest.json` change. The new
JSON requires selective force-add by main. CPU validation reran103 maintained
control/lifecycle tests, reproduced all four SQL summaries exactly read-only
(the fourth remains diagnostic), and verifies manifest/source/receipt hashes.


## Fresh limited Lookahead pair: both captures qualified, histories unchanged

The independently approved exact2 stage
`target/validation/grouped-profile-ce6d4ff-lookahead-proc-exit-20261017-01/`
**passed both captures once**, in order: new Lookahead grouped23, then new
Lookahead synchronous24. There were no retries, warmups, builds, Blocking reruns,
old20 timing/primary reruns or quality14 reruns, and no further GPU jobs.

`grouped-profile-results.json` appends `additional_profile_stages[0]` and a
`qualified_schedule_pair_index`. **Every original root field remains unchanged**:
the earlier four-profile stage still has status `failed`, all4_passed=false, its
old Lookahead pair unqualified, and its failed24 diagnostic-only. The old grouped
Lookahead23 is not substituted for the new grouped arm. Neither old profile24 nor
the earlier fixed24 profile21 is rehabilitated. The prior text and raw histories
above continue to describe their original stages.

Both new arms use identical current observer control99c6b92 (approved proc-exit
fixd20d3f8), while the qualified Blocking pair remains in its separate historical
controld48cad78 stage. Both stages use the same frozen ce6d4ff application,
archive4e15b730... and binaryd8c0bae...; **no arm is mixed across control cuts**.
The pair index identifies the exact stage/control/raw FINAL for each schedule.

### Qualified Lookahead observations

Both new reports match their own completed Lookahead primaries and each other:
25,116 logical uploads, identical ordered work/check arrays and64 exact output
fingerprints, unchanged ATOL0.00025/RTOL0.012, configuration and lifecycle-v3
quotas. Complete shared arrays remain in their original JSON locations; each new
capture binds their hashes and includes its own report/receipt provenance.

| Whole-model-process observation | Synchronous24 | Grouped23 | Grouped − synchronous / logical upload |
|---|---:|---:|---:|
| `cuStreamSynchronize` rows | 382,743 | 257,163 | −5 |
| `cudaStreamSynchronize_v3020` rows | 290 | 290 | 0 |
| Host-pin allocation rows | 2 | 2 | 0 |
| Host-pin free rows | 2 | 2 | 0 |
| H2D rows | 195,985 | 195,985 | 0 |
| H2D bytes | 82,512,273,944 | 82,512,273,944 | 0 |
| Observed kernel rows | 409,411 | 409,411 | 0 |

The stream-sync reduction is125,580, exactly5 per logical upload, matching the
predeclared hypothesis **without using it as a pass/fail threshold**. Counts per
upload are15.239011 synchronous versus10.239011 grouped. Both model processes
have the expected2 pin allocations and2 frees. All69 observed kernel-name/count
entries match; H2D/D2D/D2H counts and bytes match. Event-query and context-selection
counts may differ. Every observed API count, host sum and normalization is retained.

`cuStreamSynchronize` host sums are4.348171507s synchronous versus1.595895095s
grouped (difference−2.752276412s). These process-wide sums may overlap other work
and **are not causal elapsed savings, expert-range attribution or measured
per-upload latency**. The earlier eight unprofiled whole-test pairs and their
approximately3.9% descriptive median reduction remain unchanged and separate;
no profiled time enters that population or implies TTFT/token/request latency.
The original negative57e4629 retained pairs remain unchanged as well.

### Terminal, custody and closeout validation

Both new captures have exact identity/job/plan/launch/ancestry-bound WNOWAIT
terminal receipts, successful stopped observer acknowledgements, and final owner
wait ordering from the pinned single-authoritative-wait wrapper. Acks followed
terminal observation by1.439s grouped and1.387s synchronous, inside15s and the
original deadlines. Their recorded terminal events bind the declared model's
matching PID/birth/parent in stateZ with reasonexited; no sticky errors or
independent map/pin errors were waived. A runtime wait-call counter is not claimed.

Both supervisors exited0, without signals/timeouts, and authenticated custody
reached ECHILD with recorded owners absent. Final strict all8 idle passed three
consecutive samples; all8 local locks exited and were nonblocking-reprobed.
Offline PID/birth checks found no surviving owned jobs. Physical CUDA release
remains unobserved, not certified by idle/ECHILD/local lock receipts.

Host peak16,777,984B remains below64MiB, settled live-owner charge10MiB, cache
current/peak209,739,776/219,754,496B below320MiB, with endpoint upload/pending/
Unknown charges zero. Read-only CPU SQL re-audits reproduced both model summaries
exactly;112 maintained CPU control/lifecycle tests passed. Source/control/GO,
manifest, raw receipt and historical-preservation hashes are checked.

Evidence remains **PARTIAL_OBSERVED_MODEL_SUMMARY**: full_trace_accepted=false,
dropped_event_count=null and API exhaustiveness unproven. No expert NVTX/range,
CPU scheduling, overlap, default-switch, significance or causal-speedup claim.
Only this README, appended `grouped-profile-results.json` and manifest change;
private weights, binaries, SQLite, vendor packages and ignored raw artifacts
remain unstaged/local-only, not publicly supplied.
