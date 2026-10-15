# Claim–evidence–limit ledger

Current manuscript: **Ferrule: Bounded Storage-Backed Weight Residency**.
This ledger separates design, source-qualified factory tests, actual HTTP
observations and unobserved physical claims. Updated for the completed
`mapfixed-20261015-01` study; no GPU work is performed by this writing stage.
Historical receipts are immutable and are not repaired or pooled by later success.

## 1. Current claim map

| Claim | Evidence now available | Boundary / remaining gate |
|---|---|---|
| Bounded storage-backed residency at a selected quality-checked point | Current factory nine gates; actual 120/120 HTTP success; repeated equal logical work; host-charge maximum below policy | Not a budget frontier, total RSS/physical peak, or compact-weight residency impossibility |
| Request-time demand is observable | 1,020 anchored invocations, 120 request windows, 150 NFS-client deltas, 240 PID-I/O snapshots | Logical/application counters and shared client scope; NAS backend and startup/shutdown PID-I/O unknown |
| Same-budget lookahead has lower observed waiting/client latency | Five qualified paired blocks; every block has lower median host wait and E2E; prespecified ratios below | Not physical GPU overlap, causal decomposition, general throughput, significance or SOTA |
| Owner-local normal closure is observed | Known closed/retained owner records, zero final logical read charge, authenticated ECHILD, idle and lock reprobes | Not post-drop physical release; component faults do not close integrated E5 |
| Source-bound plans/exact completion/custody define safe reuse | Implementation mapping and Algorithms 1–3; component fixtures | Not formal verification, universal registry coverage, durable ACID or eventual reclamation |

Physical domains count disjoint resident pools once; aliases are not extra copies.
Configured accounting ceilings can overlap and omit physical consumers, and cannot
be summed into machine capacity. Physical insufficiency/necessity is an optional
stronger claim requiring compact-weight, page-cache/swap and per-pool evidence.
The current claim is **bounded residency with measured wait reduction**, not
first storage inference, nonblocking forward or performance leadership.

## 2. Immutable identities and source-cut separation

| Evidence generation | Identity / archive | Interpretation |
|---|---|---|
| Earlier factory qualification | `033519c3def098ce417e4b6ff0f141a9f30bddb1`; [qualification_v1](experiments/storage_demand/qualification_v1/README.md), commit `17b16fa` | Nine exact-once gates, distinct older source cut |
| Current factory + compiled serving source | `86c58a017d7c4b53490f536a3b1a133f2f7848cc`; 1,484 source members | Nine new same-cut factory gates; not nine repetitions of the older cut |
| Current CLI | `7cc8e0ff6e84f27090494b8e9e27dc3a388d1d2f3fc34ae39dae416e1e607a01` | Identical binary in both timing modes |
| Current constructor / composition executables | `693fe2de…24ef1f` / `0a429c47…fc8677f` | Distinct executables; common source/runtime ≠ mathematical CLI equivalence |
| Current original source archive | `e0d160996147820fcc64cfb29070b265b2297a6d9265a21de541ed77e8f2fa65` | External payload, not bundled by publication |
| Readiness-map validator / independent entry | `7d3bee2` / `ac224f7` | Collection evidence correction and new generation, not a new compiled CLI or adaptive plan |
| Completed public study | [publication_results_v1](experiments/storage_demand/paired_read_v1/publication_results_v1/README.md), `c624335` | 623 files /4,451,307 bytes =4.25 MiB; offline receipt reconstruction |
| Current plots | [analysis_v1/PLOTS.md](experiments/storage_demand/paired_read_v1/analysis_v1/PLOTS.md), `fc7f712` | 840 exact inputs; three PDFs/PNGs, deterministic data/layout/toolchain manifests |

Exact current build commands, binary/source/native/runtime pins and all nine
quality-review rows are in
[design/source-and-build.json](experiments/storage_demand/paired_read_v1/publication_results_v1/design/source-and-build.json)
and [REPRODUCE.md](experiments/storage_demand/paired_read_v1/publication_results_v1/REPRODUCE.md).
Original final-study SHA-256 is
`33459c536d5a736bb4843e80e2cf3e79016fae054e5e5664cfe859cfb089e357`;
poststudy audit is `30d123fac5be3df3eae9a0241f75897ea24a2d30cec254c41456baed85dd17c5`.
Public projection hashes are not original hashes, signatures or public availability
of omitted raw payloads. Runtime path/hash/ELF checks are not live mapped-memory
byte equivalence; overlay backing identity remains unobserved.

## 3. Current factory and actual serving quality

The current nine raw producers are accepted on `86c58a0`; original reports are
under `target/validation/fixesmetadataCLIcurrent-quality-20261015-01/`, with
report and supervisor hashes in the public build/quality projection. All nine
record natural exit, authenticated ECHILD and 12 required runtime-library pins.
Physical release remains unobserved. CUDA is 12.4.99, compiled architecture SM86,
NVTX compiled with `FERRULE_PROFILE_NVTX=0`.

| Gate family | Gates | Main metrics / common-history vocabulary values per gate | Other scope |
|---|---:|---|---|
| Full40 | 2 | 804 /1,986,560 | One per mode; prefill/first decode and diagnostic checks |
| Retained fresh | 2 | 5,427 /9,187,840 | 64 actual output fingerprints across four streams |
| Ordinary | 2 | 27 /9,187,840 | All 64 match own-schedule primary; routes/states unobserved/null |
| Cold owner | 2 | 1,809 /3,228,160 | All 28 match own current primary/config/oracle |
| Constructor | 1 | 128 finite values total | Four boundary/schedule cells ×32 values, six verified reads each |

Tolerance remains `0.00025 + 0.012*abs(reference)`, exact IDs where observed,
not universal bit-exactness. Per fresh/ordinary schedule: common history and free
feedback **each** 27 calls/9,187,840 values; reset 9/3,228,160; fork 1/248,320.
Per cold schedule: common/feedback/reset **each** 9/3,228,160; fork 1/248,320.
Fingerprints are not additional logits. Do not sum streams, observer metrics,
schedules or source cuts as a larger independent quality denominator.

Actual HTTP is separate evidence: ten completed lifecycles, **120 issued/terminal/
successful requests**, zero failed/censored/unissued; **50 measured QA**.
There are **110 exact eight-token oracle requests** and **ten two-token smoke
requests** (usage/length only). All **900 worker-enqueued token IDs** join through
worker IDs, HTTP completion IDs, ordinals, terminal usage and SSE/DONE. Enqueue is
not client consumption; positive terminal joins establish delivery here. No
client-API token-ID, retokenization, exact SSE token-time, HTTP logit/route/state
or broad task-quality claim is made.

## 4. Equal work, accounting, client I/O and closure

Both arms use PP1/TP1/EP1 Mixed Qwen3.5-35B-A3B-FP8, resident KV/lazy experts,
context128/batch8/sequence2, cohort target1/max-deferrals0 and serial requests.
Only the schedule changes; nominal routing/cache/safety/observation policy is fixed.
Both explicitly enable the real worker; the default disabled path is not a control.

| Per-lifecycle observable (all ten lifecycles) | Value / interpretation |
|---|---|
| Pipeline submitted/completed/verified operations | 40,943 each; not the number of syscalls or tokens |
| Checkpoint-reader calls | 245,658 |
| Logical reader bytes | 128,811,263,616 |
| Facade H2D bytes | 128,811,927,312; difference 663,696, not physical bus traffic |
| Artifact uploads / expert uploads / evictions | 122,829 /40,943 /40,879; recorded hits zero, not a cache-miss inference |
| Speculative reads taken | Lookahead 36,863; blocking 0 |
| Wasted read bytes / unknown read counts/bytes | All zero in the pipeline ledger |
| Logical host high-water | 16,777,984 bytes =2×3,146,112 +8,388,608 metadata +2,097,152 stack |
| Host policy | 64 MiB, including metadata8/requested-stack2; one worker/depth1/two slots |
| Device expert policy | 64 entries/320 MiB charge, including scratch8 MiB |
| Device-wide sampled maximum | 4,125 MiB in every arm; 49–54 samples/arm; policy7 GiB, not certified peak/owned GPU |
| Final state | Known owner-local closed/retained, zero logical read charges, not physical release |

Counts reconcile from 1,020 invocation rows/102 per lifecycle and 120 anchored
request windows to the owner shutdown ledger. Peaks are maxima, not deltas.
Equal totals do not provide a full route-ID signature. Host logical charge,
RSS/PSS, allocator high-water and device samples are different scopes; do not
sum independently timed maxima. OS/TLS/allocator/weak-backing headroom remains
outside the logical bound. Passive snapshots do not heal poison or refund custody.

There are **150 NFS-client deltas** (120 request +30 phase). P1 blocking's
shared-client request phase has 314,825,320 `serverreadbytes`; request windows
sum to 312,732,264. The **2,093,056-byte gap is inter-window, not attributed to
requests**. Other nine phases record zero. Shared-client counters do not establish
request-exclusive traffic, NAS cache misses or physical disk reads. Fresh owners
are not storage-cold; no cache flush occurred. **240 request PID-I/O snapshots**
support 120 PID/birth-bound deltas, but startup/shutdown PID-I/O was not collected.
Startup logical-reader bytes also remain null. No reconstruction from exit/idle.

Authenticated ECHILD, absent recorded descendants, idle and lock reprobes support
process cleanup, not device fences. Retained residency/known closure/zero logical
charges do not prove post-drop physical release. Integrated fault E5 remains open.

## 5. Prespecified paired inference, not a performance ranking

[Preregistration](experiments/storage_demand/paired_read_v1/publication_results_v1/design/preregistration.json):
five paired blocks, order B/L, B/L, L/B, L/B, B/L; ten fresh lifecycles; each arm
four initial requests +one excluded QA +two warmups +five measured QA. Absolute
7,200 s envelope, 650 s/lifecycle, 100 s final reserve, STOP/no retry/replacement,
all-five-pairs-or-null. No adaptive changes, extra pilots, cache flushes or profiling.
Elapsed study time 1,519.724531 s; launch-to-ready probes 12.847339–13.246785 s.

For each block, E2E/TTFT use the median of five measured requests. Goodput is
40 qualified tokens /sum of their client E2E durations. Aggregate is
`exp(median(log(L/B)))`. All `5^5=3125` ordered paired-block resamples are archived;
nearest ranks79/3047 give descriptive 2.5/97.5 percentile endpoints.

| Outcome | L/B | Relative change | Descriptive range |
|---|---:|---:|---|
| E2E | 0.9182346585 | −8.176534% | [0.8930346492, 0.9494606412] |
| TTFT | 0.8875530391 | −11.244696% | [0.8600389245, 0.9488801125] |
| Qualified serial-window goodput | 1.1090822447 | +10.908224% | [1.0739284940, 1.1349782205] |

All five blocks have lower observed median host demand wait (0.398–0.703 s
reductions) and client E2E (0.384–0.827 s reductions). Wait sums anchored invocation
deltas within each request, then takes the arm median. Difference of arm medians
is not median requestwise difference or an additive decomposition of client time.
The five blocks—not 50 requests or ten arms—are the analysis units. Ranges are
not coverage-guaranteed 95% CIs, significance/power results or a general SOTA claim.

The separate ten-window accounting summary is B200/193.581320 s=1.03316 tok/s,
L200/175.393210 s=1.14029 tok/s. This excludes startup/inter-request gaps, is not
concurrent server throughput, and does not replace the prespecified estimator.

**Overlap remains pending.** Both timing modes compile NVTX source annotations
with runtime profiling OFF. The separate first capture at
`experiments/storage_demand/profile_overlap_v1/artifacts-go-20261015-01/`
stops before QA on additional profiler-library allowlisting: zero target ranges,
lookahead unissued, four official warnings, full-trace acceptance false. Its
[capture-result](experiments/storage_demand/profile_overlap_v1/artifacts-go-20261015-01/capture-result.json)
records failure, authenticated drain and original idle. The independent collector
correction is ongoing; no later success, physical kernel/read overlap or timing
result is anticipated here. Host submission spans/last ticket do not prove overlap.

## 6. Historical negative evidence and separate successes

The manuscript preserves these in [Appendix A](sections/evaluation-history.tex).
Raw files, thresholds, source relations and original failures are untouched.

| Record / locator | Preserved result and boundary |
|---|---|
| [TP1 prelaunch -01](experiments/storage_demand/runs/storage-demand-tp1-middle-20261013-01/report.json) | `subprocess.DEVNULL` AttributeError before server; 4 planned/0 dispatched/4 not run; no model/OOM or reads |
| [TP1 real -02](experiments/storage_demand/runs/storage-demand-tp1-middle-20261013-02/report.json) | One DEBUG disabled-pipeline run, 4/4 HTTP, 3 text/usage canaries; Hello2 HTTP/usage only; IDs/logits/routes absent. 30 invocations/69,114 reader calls/36,240,064,128 reader bytes; H2D36,240,248,688; 11,519 uploads/11,455 evictions. Resource contract unknown, raw GPU quiescence unverified |
| [Middle quality -01](../../target/validation/storage-middle64-current-quality-20261014-01/FINAL.json) | `cuInit (3)` before inference, 0 metrics/logits; fresh/cold unissued; distinct from DEVNULL |
| [Full40 -02](../../target/validation/storage-middle64-current-quality-20261014-02/FINAL.json) | Numerical804/1,986,560 PASS; strict first-idle wrapper rejects1% utilization. Later idle does not repair it |
| [Fresh -03](../../target/validation/storage-middle64-current-quality-20261014-03/FINAL.json) | Numerical5,427/9,187,840 PASS; strict wrapper rejects6% utilization |
| [Cold -04](../../target/validation/storage-middle64-current-quality-20261014-04/FINAL.json) | 1,809/3,228,160 accepted under stable-idle v2, different3,533-input `5662b7d2…` cut; primary `same_source_cut=false` versus `fbf0c225…` |
| [Frozen CLI quality -01](../../target/validation/storage-observability-cli-quality-20261014-01/FINAL.json) | `cuInit (3)`, 0 metrics/logits/token checks; retained fresh unissued, not numerical mismatch or HTTP acceptance |
| [Earlier current-factory result](../../target/validation/expert-read-current-quality-20261014-02/FINAL.json) | Nine exact-once PASS on `033519c`; source archive `ac072e0d…d2a82f`; distinct CLI `de515995…`. Per-mode reads full40 2,089/fresh25,116/ordinary25,116/cold11,354. Host16,777,984; allocator4,106,223,616 bytes, getter may poll fences. No CLI/performance/physical release transfer |
| [Both failed paired studies](experiments/storage_demand/paired_read_v1/publication_results_v1/historical-failures.json) | **Each** 1 attempted/0 completed/9 unissued lifecycles, 120 HTTP/50 QA unissued, all five pairs retained/null. Original `artifacts-current-20261015-01`: pre-Ready TP2 accessor/prepared-source rejection. `artifacts-v3-final-20261015-01`: library-map/mode validation stop. No resume/retry/replacement/pooling |
| Independent baseline pilot | [protocol](experiments/storage_demand/newversionedprotocol/protocol.json):5 lifecycles/60 requests/25 QA planned, unexecuted |
| Independent W8A16 strict oracle | `target/validation/w8a16-full40-independent-20261002/`:1,837,493/1,986,560 logits beyond2e-4 tolerances, max_abs0.2965252995, four ordered-route mismatches; states fail; 8/8 argmax does not waive failure |
| TP2 quality selections | [model-runtime-quality](experiments/tp2_resident/model-runtime-quality.json):33 calls/35,937 metrics; full-boundary Hello/Capital2,178 fields each separate. Different `production-opts-01`37/40,293 not substituted. Test executable ≠ serving; greedy4-01 supervisor FAIL retained |
| TP2 early 001/002 | [original results](experiments/tp2_resident/RESULTS.md):Hello success, canary startup abort due PSS ESRCH /unknown nonselected1MiB; no canary HTTP |
| TP2 baseline-delta003/004 | [results](experiments/tp2_resident/RESULTS-baseline-delta.md):observer `nvidia-smi` timeout /CUDA driver initialization failure before requests; no retrospective numerical classification |
| TP2 official full profile | [audit-v2](experiments/tp2_resident/runs/tp2-official-full-profile-20261010-001/profile-audit-v2.json):incomplete collection; missing API/activity pairs; `full_trace_accepted=false`, unknown loss cause, no performance qualification |

Historical `-02` startup18.497 s, request windows51.538 s, supervisor75.874 s;
32 GPU/complete host samples: GPU4,125 MiB, PSS/RSS7,610.864/7,613.770 MiB;
last64 entries/201,351,168 resident bytes/scratch8,388,608; cache high-water219,754,496.
Its four plots retain `n=1`, DEBUG and disabled-pipeline identity, not current results.

The [TP2 resident crossblocks](experiments/tp2_resident/runs/route-batch-vllm-crossblocks-20261012-01/crossblocks-results.md)
are supplemental n=2: Ferrule14.317/14.804 versus vLLM10.376/10.330 tok/s;
vLLM has better TTFT/startup/GPU/PSS. Mixed HF-eager/GemmEx versus Marlin arithmetic,
22 GiB/device versus utilization0.8 and host budgets are unmatched;
`fairmatch=false`, `exact_speedup=null`. Actual vLLM0.28.0+cu129/Torch2.13.0+cu129
identities are not inferred from an older checkout. Earlier successful vLLM single
lifecycle startup65.7325 s, QA8≈11.31 tok/s remains a separate single-owner record.
No TP2 result proves TP1 capacity, disk necessity or universal quality.

[Startup-metadata report](../../target/validation/startup-metadata-owner-20261013-01/REPORT.md):
36e6 S1/S2 startup/PSS222.079/42.716 and139.855/44.416; F1 140.052/43.573,
40 tokens/2.877869 s=13.899 tok/s. Older e674 F1 351.114/60.149/14.227 is unpaired.
Five proc-read costs0.643/623.616/725.908/625.392/0.085 ms occur outside measured
windows; persistence overhead unmeasured; post-exit sample missing, not zero.
No causal startup/heap-release claim. Historical F32 process/adapter passes remain
profile-specific; [CURRENT_ACCEPTANCE](../../docs/CURRENT_ACCEPTANCE.md) preserves
those source cuts and earlier Mixed qualification, not current general acceptance.

## 7. Implementation and remaining gates

- [Checkpoint worker](../../crates/ferrule-model/src/checkpoint/read_pipeline.rs)
  and [source identity](../../crates/ferrule-model/src/checkpoint/source.rs):
  immutable ordered plans, real positioned reads, exact tickets, Unix ctime source
  checks. Metadata identity/descriptor hashes are not full-payload authentication.
- [Expert sequence](../../crates/ferrule-model/src/transformer/expert_reads.rs)
  and [owner split](../../crates/ferrule-model/src/transformer/standard/cuda.rs):
  canonical gate/up/down artifacts, aliases retain quota, one next read before
  compute submission, speculative errors deferred to matching demand; canceled
  reads drained/count as waste. Forward remains synchronous.
- [Registry](../../crates/ferrule-runtime/src/io/registry.rs),
  [broker](../../crates/ferrule-runtime/src/scheduling/resources.rs) and
  [architecture](../../docs/ARCHITECTURE.md): common exact-key/credit contracts do
  not prove every specialized expert/HostKV/prepared-KV path uses one registry.
  Caller cancellation does not revoke already-started irreversible physical Publish.
- Integrated E5 must test stale/delayed events, stage cancellation, cleanup failure,
  owner loss/lost ACK and retained-credit conservation. Unknown stays charged;
  no blanket recovery, bounded reclamation or formal proof is claimed.
- Full physical attribution, external fair baselines, observer overhead, long/concurrent
  workloads and alternate modalities/memory topologies remain future work.
  FlexGen/ZeRO-Inference/PowerInfer/PowerInfer-2/vLLM positioning is scoped prior art,
  not an exhaustive guarantee audit. Full metadata audit and author declarations
  remain open; no SGLang or unsupported-capability claim is inferred.

## 8. Writing-stage validation and scope

Only `main.tex`, manuscript sections, repository/paper READMEs, checklist, this ledger and
tracked `build/main.pdf` are authored. Figures, bibliography, Makefile, experiments,
raw receipts and production code are unchanged. Other-agent `profile_overlap_v1`
work is neither staged nor treated as completed evidence.

Actual CPU-only validation: portable checker +29 tests pass on Python3.10.9 and
3.12.4; analysis16 tests pass; byte-exact plot `--check` passes all outputs and840
input hashes; finite `make -C papers/arxiv` passes with zero converged warnings,
box/reference/citation issues. The [README](README.md#manuscript-validation-and-edit-scope)
records final PDF pages/render review. A local protected-input manifest covers919
preexisting assets/archive files, checked before staging. No GPU/HTTP/experiment
rerun or new scientific acceptance is implied by these checks.
