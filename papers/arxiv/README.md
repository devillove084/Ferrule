# Ferrule manuscript workspace

**Ferrule: Bounded Storage-Backed Weight Residency** — local English working
draft, not a submission. Author: Lu Huanbing, Independent Researcher,
`luhuanbing084@gmail.com`. Unprovided ORCID, funding and declarations are not inferred.

## Current result: qualified bounded residency and measured wait reduction

The evaluation now follows **current factory quality → actual HTTP
quality/budgets → equal logical work and scoped I/O → paired scheduling →
fault-contract limits**, rather than the chronology of earlier attempts.

The completed `mapfixed-20261015-01` study has five preregistered paired blocks,
ten fresh lifecycles, **120/120 successful requests** and 50 measured QA requests.
Both modes use the identical CLI SHA-256
`7cc8e0ff6e84f27090494b8e9e27dc3a388d1d2f3fc34ae39dae416e1e607a01`,
built from `86c58a017d7c4b53490f536a3b1a133f2f7848cc` (1,484 source members).
The readiness-map validator is from `7d3bee2`; the separate generation entry is
`ac224f7`. Neither changes the compiled CLI or adaptively changes the study plan.
The earlier `033519c` nine-gate qualification and the current `86c58a0` nine gates
are separate source cuts, **not 18 repetitions or pooled quality evidence**.

| Paired outcome | Aggregate lookahead/blocking | Relative change | Descriptive range |
|---|---:|---:|---:|
| Client E2E | 0.9182346585 | 8.18% lower | [0.8930346492, 0.9494606412] |
| Client TTFT | 0.8875530391 | 11.24% lower | [0.8600389245, 0.9488801125] |
| Qualified serial-window goodput | 1.1090822447 | 10.91% higher | [1.0739284940, 1.1349782205] |

The estimator is `exp(median(log(L/B)))` over **five paired blocks**. E2E/TTFT
use each arm's median of five measured QA requests. Goodput is 40 qualified
completion tokens divided by the sum of those five E2E durations. All `5^5 = 3125`
ordered block resamples are enumerated; nearest ranks 79/3047 give descriptive
2.5/97.5 percentile ranges, **not coverage-guaranteed 95% confidence intervals,
p-values, power or SOTA evidence**. Requests are not independent replicates.

Across the ten measured arm windows, blocking totals 200 tokens/193.581 s
(1.03316 tok/s), lookahead 200/175.393 s (1.14029 tok/s). These sums are a separate
serial-window accounting summary, not the paired estimator or general server
throughput. Startup and inter-request gaps are excluded. Order is B/L, B/L, L/B,
L/B, B/L; there are no retries, replacements, extra warmups, cache flushes or
profiled timings. Both modes compile NVTX but run with profiling **OFF**.

## Quality, work, memory and custody scopes

- **Factory:** current nine gates pass once: constructor plus both schedules'
  full40, retained-fresh, ordinary and cold-owner paths. Per gate, full40 is
  804 metrics/1,986,560 vocab values; fresh 5,427/9,187,840; ordinary
  27/9,187,840 with all 64 fingerprints matching its own diagnostic primary;
  cold 1,809/3,228,160 with all 28 matching its own current primary. Constructor
  has four cells/128 finite values, six verified reads per cell. Stream
  denominators stay separate; ordinary routes/states remain null.
- **HTTP:** 110 exact eight-token oracle requests plus ten two-token usage/length
  smoke requests. All 900 actual worker-enqueued token IDs are joined to HTTP
  IDs, ordinals, terminal usage and SSE/DONE. These are not client-API token IDs,
  retokenized text or exact SSE token timestamps. Enqueue and client consumption
  remain distinct; positive terminal joins establish delivery here.
- **Equal work per lifecycle:** 40,943 submitted/verified pipeline operations,
  245,658 checkpoint-reader calls, 128,811,263,616 logical reader bytes and
  128,811,927,312 facade H2D bytes. Difference: 663,696 bytes, not physical bus
  traffic. Lookahead takes 36,863 speculative reads versus zero for blocking;
  both have zero wasted bytes and unknown read counts/bytes. Nominal route/cache
  policy is unchanged; no full route-ID signature or cache-miss inference.
- **Budgets:** same worker/depth-one queue/two slots; 64 MiB logical host limit
  includes 8 MiB metadata and 2 MiB requested stack. Peak logical host charge is
  16,777,984 bytes. Device cache is 64 entries/320 MiB including 8 MiB scratch.
  Device-wide sampled maximum is 4,125 MiB against a 7 GiB policy in every arm,
  not a certified continuous peak or exact process-owned memory. Do not add
  PSS, GPU samples or independently timed logical maxima.
- **I/O:** 1,020 invocation rows, 150 NFS-client deltas, 240 request PID-I/O
  snapshots. P1 blocking's shared-client request phase is 314,825,320 bytes;
  request windows sum to 312,732,264. The 2,093,056-byte inter-window gap is not
  assigned to requests. Other phases record zero, not proof of no disk I/O.
  Startup/shutdown PID-I/O and startup logical-reader bytes are missing/null.
  Fresh owners are not storage-cold; NAS backend traffic remains unknown.
- **Closure:** all ten owners are recorded known closed/retained with zero
  final logical read charges, authenticated ECHILD, idle and lock-reprobe
  evidence. This is not post-drop physical release. Component fault fixtures
  do not close integrated E5.

**RQ status:** RQ1 has a qualified controlled-budget point, not a capacity
frontier/total physical bound; RQ2 closes logical/shared-client observability,
not backend attribution; RQ3 reports measured paired wait/client outcomes,
not GPU overlap/causality; RQ4 has owner-local/process closure but integrated
faults and physical release remain open. Physical-memory necessity, nonblocking
forward, universal registry integration and performance leadership are unclaimed.

The separate first profiling capture stopped before QA on an additional
profiler-library allowlist requirement: zero target ranges, lookahead unissued,
four official warnings and full-trace acceptance false. Independent collector
work is ongoing and is **not anticipated as a result or included in this commit**.

## Figures, tables and retained history

The title, three standard algorithms and three colored mechanism assets remain.
Algorithm 3 states the actual one-next-expert schedule and alias-aware budget.
Current figures are included at full text width:

| Asset under `figures/` | Main-text label / role |
|---|---|
| `paired-read-accounting.pdf` | `f:paired-accounting`: equal work and distinct accounting/sample scopes |
| `paired-read-latency.pdf` | `f:paired-latency`: five paired blocks, exact estimator and descriptive ranges |
| `paired-read-read-wait.pdf` | `f:paired-read-wait`: less host waiting associated with lower client E2E |

Main tables report current factory qualification, shared controls, per-lifecycle
work, paired outcomes and claim closure. Appendix A retains all four historical
`storage-demand-*.pdf` plots unchanged, their equations and disabled-pipeline
`-02` identity; the two failed paired studies; earlier numerical/wrapper failures;
and separate TP2/vLLM/startup observations. Neither failed study is pooled:
**each has 1 attempted/0 completed lifecycles, 120 unissued HTTP requests,
50 unissued QA slots and five null pairs**. Historical vLLM comparisons remain
unmatched supplements, not SOTA results. Failed W8A16 quality is retained.

## Portable artifacts and CPU-only reproduction

- [Completed study](experiments/storage_demand/paired_read_v1/publication_results_v1/README.md),
  commit `c624335`: **623 files / 4,451,307 bytes (4.25 MiB)**. This corrects the
  earlier decimal/binary shorthand “4.45 MiB.” Python 3.10+ standard-library
  offline reconstruction; external executable/model/oracle/source/native
  payloads and private authentication are not bundled or replayed.
- [Current plot guide](experiments/storage_demand/paired_read_v1/analysis_v1/PLOTS.md),
  commit `fc7f712`: 840 hash-bound inputs, deterministic data/rendering manifests,
  three PDFs/PNGs, 16 tests. Byte reproducibility is scoped to the pinned renderer.
- [Earlier factory archive](experiments/storage_demand/qualification_v1/README.md),
  commit `17b16fa`: `033519c` only; 249 files/5.38 MiB, 37 offline tests.
  It is not the current CLI's qualification archive.

From repository root:

```sh
make -C papers/arxiv
python3 -S -B papers/arxiv/experiments/storage_demand/paired_read_v1/publication_results_v1/check.py
python3 -S -B papers/arxiv/experiments/storage_demand/paired_read_v1/publication_results_v1/test_publication.py
python3 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/paired_read_v1/analysis_v1/tests -v
/home/admin/wxw403883/anaconda3/bin/python3.12 -B papers/arxiv/experiments/storage_demand/paired_read_v1/analysis_v1/plot_paired_read.py --check
```

The last path identifies the local pinned plotting environment; see the plot
guide for toolchain/font hashes. The portable study checker needs no such path,
GPU, network, model or Ferrule import. Do not replay consumed experiment launch
commands or overwrite sealed generations.

## Manuscript validation and edit scope

This stage changes only manuscript sources, this README, the repository README,
[checklist](PAPER_COMPLETION_CHECKLIST.md), [evidence ledger](EVIDENCE_LEDGER.md)
and tracked `build/main.pdf`. Figures, bibliography, Makefile, production code,
experiments and raw receipts are untouched. Local build/review outputs are not
staged. No GPU query, launch, cache change or experiment was performed.

- Finite `make -C papers/arxiv` (120 s limit): **PASS**, current PDF **36 pages**
  (29 main/reference pages + 7 appendix pages), three algorithms and ten figures.
- Converged LaTeX log: **zero** warnings, undefined references/citations,
  overfull/underfull boxes, oversized floats or duplicate labels.
- Portable checker and **29/29 tests PASS on Python 3.10.9 and 3.12.4**.
- Current analysis **16/16 tests PASS**; byte-exact `--check` **PASS** for all
  current plot outputs and 840 input hashes.
- Ghostscript rendered all **36 pages** in color (110 dpi) and grayscale
  (90 dpi). All-page color contact sheets, full-size Algorithm 3 (p.15), factory
  tables (p.20), estimator (p.22), and all three current plots/captions (pp.23–25)
  were reviewed; current figures were also reviewed in grayscale. No clipping,
  caption occlusion or incomplete algorithm was observed. Mechanism colors stay
  unchanged; the four historical plots remain on pp.32–35.
- All **919 protected asset/archive files** retain their hashes; source references,
  local documentation links and `git diff --check` pass. The final PDF hash,
  source hashes and validation scope are in local `build/paired-paper-review.json`.
  Building and reading receipts are not new numerical, fault, overlap or submission
  acceptance.

The tracked PDF is [build/main.pdf](build/main.pdf). Final author declarations,
full related-work metadata audit, external-payload recovery and submission
approval remain separate tasks; see the checklist. Only explicit authored paths
and the tracked PDF belong in the manuscript commit, never another agent's
profiling directory or the entire workspace.
