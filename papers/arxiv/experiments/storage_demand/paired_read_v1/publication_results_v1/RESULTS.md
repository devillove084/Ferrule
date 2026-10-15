# Public results: completed paired HTTP read-schedule study

Source generation: `mapfixed-20261015-01`; immutable study date: **2026-10-15**.

Generated from all archived observations by `seal.py` / `check.py`; not hand-entered outcomes.

## Prespecified paired results

Ratios are Lookahead / Blocking. Each block uses its arm median of five measured QA
requests for E2E and TTFT; goodput is 40 qualified generated tokens divided by
the sum of those five client E2E durations. The aggregate is the exponentiated
median of the five block log ratios.

| Metric | Ratio | Relative change | Descriptive interval |
|---|---:|---:|---|
| Client E2E | 0.9182346585 | -8.176534% | [0.8930346492, 0.9494606412] |
| Client TTFT | 0.8875530391 | -11.244696% | [0.8600389245, 0.9488801125] |
| Qualified serial-window goodput | 1.1090822447 | +10.908224% | [1.0739284940, 1.1349782205] |

All **3,125** ordered five-block bootstrap resamples are archived in
`bootstrap-enumeration.json.gz`. Endpoints use nearest ranks 79 and 3047.
This is exploratory/descriptive n=5, **not a guaranteed-coverage CI**, significance
test, power calculation, or general serving-performance claim.

| Block | Order | Blocking E2E median (s) | Lookahead E2E median (s) | E2E ratio |
|---|---|---:|---:|---:|
| P1 | blocking then lookahead | 7.760480475 | 7.228216227 | 0.9314134931 |
| P2 | blocking then lookahead | 7.732933735 | 6.905777765 | 0.8930346492 |
| P3 | lookahead then blocking | 7.594348919 | 7.210535394 | 0.9494606412 |
| P4 | lookahead then blocking | 7.965932962 | 7.163044952 | 0.8992097958 |
| P5 | blocking then lookahead | 7.709717179 | 7.079329520 | 0.9182346585 |

## Denominators and evidence

- 5 complete pairs, 10 completed lifecycles; 120/120 issued, terminal and successful requests.
- 50/50 measured QA; 110 exact eight-token oracle matches and 10 two-token length-only smoke requests.
- 900 actual worker-generated tokens joined to HTTP completion IDs, terminal usage and SSE/DONE.
- 1,020 decoder invocation records; 150 NFS-client deltas (120 request + 30 phase).
- 240 PID/birth-bound request I/O snapshots; 20 library snapshots with 12/12 required pins each.
- No retries, replacements, pooled historical attempts, extra pilot lifecycles, or profiling timings.
- Recorded study elapsed time: 1519.724530699 s.

| Arm | Owner read bytes (logical) | Demand wait (s) | Speculative reads | Wasted read bytes | Peak logical host charge (bytes) |
|---|---:|---:|---:|---:|---:|
| P1-blocking | 128811263616 | 35.143921595 | 0 | 0 | 16777984 |
| P1-lookahead | 128811263616 | 29.123095646 | 36863 | 0 | 16777984 |
| P2-blocking | 128811263616 | 33.394299416 | 0 | 0 | 16777984 |
| P2-lookahead | 128811263616 | 27.860419773 | 36863 | 0 | 16777984 |
| P3-lookahead | 128811263616 | 28.682591563 | 36863 | 0 | 16777984 |
| P3-blocking | 128811263616 | 33.533046963 | 0 | 0 | 16777984 |
| P4-lookahead | 128811263616 | 28.645968189 | 36863 | 0 | 16777984 |
| P4-blocking | 128811263616 | 33.913550177 | 0 | 0 | 16777984 |
| P5-blocking | 128811263616 | 33.674756412 | 0 | 0 | 16777984 |
| P5-lookahead | 128811263616 | 28.161872055 | 36863 | 0 | 16777984 |

Read totals, waits and charge are owner-local logical/host observations, not NAS backend traffic or GPU overlap.
The final logical read charge is zero in every arm; owner-local resources are known closed
and retained. That is **not** a post-drop physical-release certificate.

## Separate historical failures

Each earlier study (`artifacts-current-20261015-01` and `artifacts-v3-final-20261015-01`)
had 1 attempted lifecycle, 0 completed lifecycles, 120 unissued requests, and 50 unissued
measured-QA slots. Their five-pair aggregates remain null. Neither is pooled with this study.

## Outstanding claim limits

- Exploratory n=5 exact 3125-draw paired-block descriptive bootstrap; no significance, coverage guarantee, power or SOTA claim.
- No pooling with either prior failed study. No replacements, retries, extra warmups, CUDA probes, cache flushes or profiling.
- Actual IDs are worker enqueue records joined to HTTP IDs, not retokenized text; SSE content events are not token boundaries.
- Pipeline last_receipt is not a complete ticket journal or GPU-overlap proof; host wait intervals are not CUDA timelines.
- Logical checkpoint/source checks and facade H2D are distinct from NFS client counters and physical backend traffic.
- Expert upload/hit events and decoder row counts are reported, not a full expert-route-ID signature or cache miss inference.
- NFS counters include shared mount traffic; fresh servers do not ensure cold storage/page cache.
- Required file path/hash/ELF-role checks do not establish mapped-memory byte equivalence; overlay backing identity is unobserved.
- Physical process-GPU byte attribution remains unknown; device-exclusive sampled memory is not exact owned-process accounting.
- Zero logical charges and known owner-local closed/retained status do not prove post-drop or universal physical resource release.
- Requested startup/shutdown PID-birth-bound /proc/io phase deltas were not collected by the reused runner; unobserved, not reconstructable after process exit. Request-boundary /proc/io and startup/request/shutdown NFS mount counters are present.
