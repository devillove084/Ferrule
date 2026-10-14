# Allocator V3 actual smoke and crossed A/B closeout

Status: completed_descriptive_only. Actual S1/S2 passed and were promoted; both crossed cycles passed. Six fresh authenticated lifecycles, no retries. No production or paper adapter changes this GPU round. No official profile or foreign process kill.

## Measured QA8

Each A/B lifecycle: excluded first + two warmups = 3 unmeasured QA8, then 5 measured serial QA8; every request actually 5 input / 8 output tokens with matching canary. Across four lifecycles: 12 unmeasured + 20 measured requests, 160 measured output tokens.

| Cycle/order | Artifact | Startup s | Mean TTFT s | Mean E2E s | Usage output tok/s | Measured wall s |
|---|---|---:|---:|---:|---:|---:|
| C1 BEFORE → AFTER | BEFORE | 349.615 | 0.529517 | 1.438295 | 5.402126 | 7.404493 |
| C1 BEFORE → AFTER | AFTER | 360.768 | 0.490692 | 1.336879 | 5.821680 | 6.870869 |
| C2 AFTER → BEFORE | AFTER | 366.398 | 0.498628 | 1.367931 | 5.664873 | 7.061059 |
| C2 AFTER → BEFORE | BEFORE | 366.180 | 0.542246 | 1.478023 | 5.201363 | 7.690292 |

Observed AFTER rate difference: C1 +7.766%, C2 +8.911%; TTFT mean -7.332%/-8.044%, E2E mean -7.051%/-7.449%. Two crossed pairs are descriptive only, not significance, isolated allocator causality, fairmatch or an exact speedup claim. Rate denominator is the recorded fixed-work HTTP window including client terminal overhead, not pure decode. Actual usage tokens are not exact visible token IDs.

## Allocator and smoke

A/B AFTER, each owner 0 and 1 in BOTH cycles: ready batches/fences = 1/3; shutdown = 108149/324447; boundary delta = 108148/324444. Counts span the lifecycle requests between ready and shutdown, not the five measured requests alone. Fences are successful complete created event sets, not total CUDA API/event calls. BEFORE has no corresponding fields; they remain unavailable/null, not zero.

Every ready/shutdown boundary across six lifecycles: pending retirement zero, arena backing 3 GiB/owner, allocation failures zero, no logged ordinary retirement pressure or quarantine. Pending/Unknown peaks remain null. Full observed raw metrics and cache snapshots are retained in lifecycle-audit.json.

| Smoke | Startup s | Cancellation/reuse | ECHILD / post-idle |
|---|---:|---|---|
| S1 | 348.456 | verified | true / verified |
| S2 | 360.959 | verified | true / verified |

## Scope and custody

- BEFORE: 9ef9841598b4694fb611250bf8876c4314d252400136ddb23f606a979cadb656, frozen historical source/build/quality receipts. AFTER: 7e1c23a88143cb6455e3c6e10bfe43a567fc97b5fd45745800eae3a1e39a0e73, exact registered V3 release. source/build/runtime/review pins and production delta are in build-source-scope.json.
- Same TP2 thread, devices 0/1, 3 GiB arena/owner within 4 GiB workspace, 22 GiB/device cap, ctx128/batch8/seq2, resident/no-offload. A/B info-only and rankperf off. S1/S2 retain the prepared request-DEBUG exception for cancellation proof, not rank performance logging.
- Sampled process CUDA_VISIBLE_DEVICES and NVIDIA_VISIBLE_DEVICES match the two selected physical UUIDs and observer selection throughout recorded diagnostics.
- All 6 finite authenticated supervisors: exit 0, not timed out, proven_drained, ECHILD=true; normal owner shutdown; selected locks released. GPU idle is independently verified by the eight-device observer, NOT inferred from supervisor exit.
- Initial and final eight-card idle: all 0 MiB, utilization 0, no listed apps. All eight locks were nonblockingly acquired/validated/released before execution and reacquired/released after. These probes are not a claim of holding all eight locks continuously; each lifecycle used the existing selected-UUID locks.
- A/B sampled device peak 22593667072 bytes = 21.042 GiB/card, within 22 GiB cap. Sampling is not continuous/exact peak proof. NVML host-to-local PID attribution is unknown; no claim that foreign interference is proven absent. No foreign processes were signalled.

## Evidence and next phase

owner-closeout.json links actual smoke, promoted admission, both cycle summaries, four measurements, six lifecycle audits, build/source scope and 241 raw/admission file hashes. Every launch/promote/admission/prepare command has its argv/cwd/exit/log receipt here. No CPU fixture repair was required this round; previously sealed workflow source pins remain unchanged.

Actual run directories: ../allocator-v3-serving-smoke-20261011-01, ../allocator-v3-crossed-c1-20261011-01, ../allocator-v3-crossed-c2-20261011-01. Raw server logs, HTTP/SSE, samples, ready/shutdown snapshots and custody receipts remain in place.

Official profiling was NOT run. The generic profile-admission.json emitted by existing promotion is not the fixed info-only retirement profile policy and is not authorized for launch by this closeout. The next separately authorized phase must use retirement_profile.CONFIG/admission for the same 9ef/7e1c artifacts and preserve actual HTTP anchors, UUID/PID correlation and trace completeness caveats. No probe grant or profile launch is issued here.
