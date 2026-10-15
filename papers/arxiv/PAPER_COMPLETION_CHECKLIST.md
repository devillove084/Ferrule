# Ferrule manuscript completion checklist

**Ferrule: Bounded Storage-Backed Weight Residency**
Lu Huanbing · Independent Researcher · `luhuanbing084@gmail.com`

Updated for the completed `mapfixed-20261015-01` study. Checked items identify
specific evidence, not publication readiness or support for every Ferrule path.
Historical failures and stronger parked claims remain below. The governing
scope is **bounded accounting plus measured wait reduction**, not proof that
compact weights exceed total physical memory, GPU overlap, or external SOTA.

## 1. Submission-core status

### RQ1 — current quality and a controlled-budget completion point

- [x] Bind the selected PP1/TP1/EP1 Mixed profile, representation, scheduler,
  model/tokenizer/oracles and compiled CLI. Current source `86c58a0`, CLI
  `7cc8e0ff…e607a01`, 1,484 source members; both modes use this same executable.
- [x] Current nine-gate factory qualification: constructor plus blocking/lookahead
  full40, retained-fresh, ordinary and cold. Keep each stream, observer mode,
  executable and schedule denominator separate. Ordinary routes/states are null.
- [x] Actual serving acceptance: ten completed lifecycles; 120/120 successful
  requests; 110 exact eight-token oracle checks and ten two-token smoke checks.
  All 900 worker-enqueued IDs join to HTTP IDs, terminal usage and SSE/DONE.
  This does not claim client-API token IDs or serving logits/routes/states.
- [x] Same logical host/device policies: 64 MiB host includes metadata 8 MiB and
  requested stack 2 MiB; 64-entry/320 MiB expert cache includes 8 MiB scratch.
  Observed host high-water 16,777,984 bytes; device-wide sampled max 4,125 MiB
  against a 7 GiB policy. Samples, RSS/PSS and logical charges are not summed.
- [ ] Capacity frontier/budget sweep, total physical working-set attribution and
  continuous peaks. The current point does not close these stronger questions.
- [ ] Compact-weight residency impossibility/physical-memory necessity, **only
  if restored as a claim**. Cache misses or a small configured cache do not prove it.

Evidence: [public build/quality identity](experiments/storage_demand/paired_read_v1/publication_results_v1/design/source-and-build.json),
[results](experiments/storage_demand/paired_read_v1/publication_results_v1/RESULTS.md),
[main evaluation](sections/evaluation.tex).
The older `033519c` factory nine gates belong to
[qualification_v1](experiments/storage_demand/qualification_v1/README.md), not
18 pooled runs or mathematical CLI equivalence.

### RQ2 — aligned logical demand and shared-client observations

- [x] Reconcile 1,020 decoder invocations/120 request windows and owner totals.
  Every lifecycle: 40,943 submitted/verified operations; 245,658 reader calls;
  128,811,263,616 reader bytes; 128,811,927,312 facade H2D bytes; zero waste and
  unknown read counts/bytes. These are logical counters, not disk/bus bytes.
- [x] Preserve all 150 shared NFS-client deltas (120 request +30 phase) and
  240 PID/birth-bound request I/O snapshots. P1 blocking phase 314,825,320 bytes;
  request windows 312,732,264; gap 2,093,056 not assigned to requests.
- [x] Record zeros in the other nine request phases without inferring no disk
  I/O, cold storage or request-exclusive NAS attribution. Fresh server ≠ cold cache.
- [x] Disclose missing startup/shutdown PID-I/O and startup logical-reader data
  as null/unobserved. No reconstruction from exited processes.
- [ ] Startup/shutdown PID-I/O collection, request-exclusive physical attribution,
  NAS backend block/network traffic, cache/swap residency and sensitivity sweeps.
  These would be new measurements, not repairs of immutable missing fields.

The scoped logical/shared-client question is answered; backend physical traffic
remains unknown. See [accounting plot](figures/paired-read-accounting.pdf).

### RQ3 — same-budget paired scheduling outcomes

- [x] Both controls use the real worker/depth-one queue/two charged slots;
  default disabled pipeline is not the control. Lookahead submits one next read
  after current upload/before compute submission; whole forwards remain synchronous.
- [x] Current factory and HTTP qualification, identical nominal route/cache/safety
  policies, observation settings and CLI. Only `--expert-read-schedule` differs.
- [x] Preregistered independent generation: validator `7d3bee2`, entry `ac224f7`,
  five paired blocks, order B/L, B/L, L/B, L/B, B/L; 7,200 s absolute envelope,
  650 s/lifecycle and 100 s final reserve. No adaptive changes, retries,
  replacements, extra warmups/pilots, cache flushes or profiling timings.
- [x] All five pairs qualify. Preserve 120 total/50 measured-QA denominators;
  requests are nested observations, not independent statistical replicates.
- [x] Exact estimator `exp(median(log(L/B)))`: E2E 0.9182346585, TTFT
  0.8875530391, qualified serial-window goodput 1.1090822447. Enumerate all
  3,125 ordered block resamples; ranks 79/3047 are **descriptive** endpoints,
  not coverage-guaranteed 95% CIs, p-values, power or a SOTA test.
- [x] All five blocks show lower host wait and E2E. Lookahead takes 36,863
  speculative reads/lifecycle versus zero; observed total work is equal and
  wasted bytes zero. This is not a causal critical-path decomposition.
- [x] Separate the ten-window total-work summary from the paired estimator:
  B 200 tokens/193.581 s, L 200/175.393 s; neither is general server throughput.
- [ ] Physical kernel/read/copy overlap. Both timing arms compile NVTX source
  annotations with runtime profiling OFF. First separate capture stopped before
  QA on additional profiler-library allowlisting: zero target ranges, lookahead
  unissued, four official warnings, `full_trace_accepted=false`. Independent
  collector correction remains ongoing, not an anticipated result.
- [ ] Longer workloads, concurrent serving, tails, independent study repetition,
  instrumentation cost, budget/cache sensitivity and stronger causal tests.

Evidence: [preregistration](experiments/storage_demand/paired_read_v1/publication_results_v1/design/preregistration.json),
[portable results](experiments/storage_demand/paired_read_v1/publication_results_v1/RESULTS.md),
[plot definitions](experiments/storage_demand/paired_read_v1/analysis_v1/PLOTS.md).

### RQ4 — custody, closure and faults

- [x] Distinguish known owner-local closed/retained state and zero logical read
  charges from post-drop physical release. All ten lifecycle process drains,
  authenticated ECHILD, idle and lock-reprobe observations are recorded.
- [x] Map component source/ticket/alias/speculative-error/cancel/drain fixtures
  to the contract without claiming integrated HTTP fault coverage.
- [ ] **Integrated E5:** delayed/stale completion, cancel at each stage,
  primary-plus-cleanup failure, owner loss, lost ACKs, original versus repeated
  submissions, publication/reuse and retained-credit conservation.
- [ ] Post-drop physical release and named device quiescence evidence. Host join,
  process exit, zero accounting and idle are not device fences. Unknown must stay
  charged/owned; no fabricated rollback, poison healing or eventual reclamation.

## 2. Artifact and manuscript checks

- [x] Current public receipt archive committed at `c624335`: 623 files,
  **4,451,307 bytes /4.25 MiB** (not 4.45 MiB), offline standard-library checker.
  This stage reruns checker and 29/29 tests on Python 3.10.9 and 3.12.4.
- [x] Current plots committed at `fc7f712`: 840 exact input hashes; three PDFs
  and PNGs; deterministic data/layout/renderer manifests. This stage passes
  16/16 analysis tests and read-only byte-exact `--check`.
- [x] Earlier `033519c` receipt archive `17b16fa`: 249 files/5.38 MiB, historical
  checker/37-test acceptance. No new GPU qualification inferred from curation.
- [x] Main narrative prioritizes current evidence; includes three actual plots,
  current qualification/configuration/work/outcome/claim tables and Algorithm 3.
  Four historical `-02` plots move intact to Appendix A with failed studies and
  the unmatched TP2/vLLM supplement. Mechanism colors remain unchanged.
- [x] Finite LaTeX build passes with zero final warnings or box/reference issues.
  Current PDF/review details are in [README](README.md#manuscript-validation-and-edit-scope).
- [ ] Independent final submission/visual acceptance, full primary-source
  citation/guarantee audit and author declarations. Local build is not submission.
- [ ] External executable/model/oracle/native/source payload recovery and authorized
  rerun. Public hash pointers are not payload availability, signatures, mapped-memory
  equivalence or replay of private authentication. Never replay consumed generations.

## 3. Historical outcomes retained, never pooled

See [Appendix A](sections/evaluation-history.tex) and
[EVIDENCE_LEDGER.md](EVIDENCE_LEDGER.md) for identities and receipts.

| Record | Preserved outcome / denominator |
|---|---|
| TP1 prelaunch `-01` | Missing `subprocess.DEVNULL`; 4 planned/0 dispatched/4 not run; no model/read result |
| TP1 real `-02` | One DEBUG disabled-pipeline run; 4/4 HTTP, 3 limited text/usage canaries; Hello2 usage-only, IDs absent; 30 invocations/36,240,064,128 reader bytes; final resource contract unknown |
| Middle quality `-01` | `cuInit (3)` before inference; 0 metrics/logits, fresh/cold unissued |
| Full40 `-02`, fresh `-03` | Numerical PASS; original strict post-idle wrappers FAIL (1%/6% utilization); later idle does not repair them |
| Cold `-04` | Accepted under stable-idle v2 on different `5662b7d2…` cut; old primary `same_source_cut=false` |
| Frozen CLI quality `-01` | `cuInit (3)`, 0 metrics/logits/token checks; retained fresh unissued |
| Original enabled paired study | Pre-Ready prepared-source/TP2 accessor failure; 1 attempted/0 completed, 120 HTTP/50 QA unissued, all five pairs null |
| Corrective V3 first execution | Loaded-library/mode validation stop; independently 1 attempted/0 completed, 120 HTTP/50 QA unissued, five null pairs |
| Independent baseline pilot | 5 planned lifecycles/60 requests/25 QA, unexecuted; not added to paired study |
| W8A16 independent full40 | Strict logits/routes/states FAIL; 8/8 argmax agreement does not waive failure |
| TP2 greedy4-01 | Supervisor failure retained despite later greedy4-02 closure |
| TP2 early serving 001/002 | Hello successes, canary startup aborted (PSS ESRCH / unknown nonselected 1 MiB); no canary HTTP |
| TP2 baseline-delta 003/004 | Observer `nvidia-smi` timeout / CUDA driver initialization failure before requests |
| TP2 official full profile | Incomplete collection and missing API/activity relations; full trace rejected, no performance acceptance |

## 4. Conditional follow-up, not prerequisites for the scoped result

<a id="parked-broad-scope"></a>

The former broad E2/vLLM engineering plan is parked, not silently marked passed.
It proposed a TP1 `p128-o32-c1` two-block pilot followed by at least five fresh
paired blocks, 8 warmups/100 measured requests per run, and a frozen 10,000-draw
paired bootstrap. Its zero-retreat criterion required throughput ratio and CI
lower bound ≥1 (strict advantage lower bound >1), with common quality/work/budgets.
That **different** confirmatory plan is not the present five-block descriptive
within-Ferrule experiment; n=5 alone never guarantees power or coverage.

- [ ] Fair external controls and physical-budget/quality matching before any
  external speedup, parity or SOTA claim. Current TP2 `fairmatch=false` and
  `exact_speedup=null` remain; vLLM is supplemental, not an unconditional gate.
- [ ] Capacity/sensitivity matrix with retained failed, infeasible, OOM, censored
  and unknown points. Never turn infeasibility into zero time/infinite speedup.
- [ ] Normal-path and observation overhead, safe ablations, replayable fault
  traces and exact clock/API/activity correlation if claiming physical overlap.
- [ ] Multimodal, shared/unified-memory, other precision/topology/backend support:
  independent full-output quality/resource contracts, not transfer from this profile.
- [ ] Primary metadata/citation audit for FlexGen, ZeRO-Inference, PowerInfer,
  PowerInfer-2 and vLLM; SGLang remains unaudited. No unsupported-capability inference.
- [ ] ORCID, funding, acknowledgments, conflicts and artifact/third-party licensing:
  only author-confirmed values, never invented “none.”

A new claim or protocol needs fresh preregistration, output destinations and
owner authorization. This writing stage does not launch GPU work or change any
experiment, figure, production source or sealed historical record.
