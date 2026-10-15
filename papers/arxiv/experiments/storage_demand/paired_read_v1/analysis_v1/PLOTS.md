# Paired-read figures: public CPU regeneration

`render_public.py` is the default reproduction path. It reads only the checked
`publication_results_v1` archive and shared plotting source; no private experiment
tree, GPU, workload launch, model, NAS access, or network is needed. Archived
measurements and their frozen manifest are never modified or resealed.

## Exact captions and suggested LaTeX labels

### latency

PDF: `papers/arxiv/figures/paired-read-latency.pdf`
PNG: `papers/arxiv/figures/paired-read-latency.png`

Client latency and quality-qualified goodput in the sealed five-block paired-read study. Each connected arm value is the median of five measured-QA requests; the independent resampling unit is the paired block (n=5), not the 50 requests. B (blue circles) is blocking and L (teal squares) is lookahead. Rows retain the frozen execution orders P1 B/L, P2 B/L, P3 L/B, P4 L/B, P5 B/L. The log-scale ratio panel shows each block (circles B/L, triangles L/B), unity, and the aggregate exp(median log(L/B)) as a diamond. E2E, TTFT, and goodput ratios are 0.91823, 0.88755, and 1.10908. Gray strips and bracketed numbers are descriptive nearest-rank 2.5/97.5 percentile ranges from all 3,125 ordered paired-block bootstrap resamples (ranks 79 and 3047), not coverage-guaranteed confidence intervals or significance tests. Goodput is 40 actual quality-qualified completion tokens divided by the sum of five client E2E durations per arm. Startup launch-to-ready probe times are separate and excluded from request latency. This single instrumented study supports no repetition-based confidence, power, or SOTA claim.

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/paired-read-latency.pdf}
  \caption{Client latency and quality-qualified goodput in the sealed five-block paired-read study. Each connected arm value is the median of five measured-QA requests; the independent resampling unit is the paired block (n=5), not the 50 requests. B (blue circles) is blocking and L (teal squares) is lookahead. Rows retain the frozen execution orders P1 B/L, P2 B/L, P3 L/B, P4 L/B, P5 B/L. The log-scale ratio panel shows each block (circles B/L, triangles L/B), unity, and the aggregate exp(median log(L/B)) as a diamond. E2E, TTFT, and goodput ratios are 0.91823, 0.88755, and 1.10908. Gray strips and bracketed numbers are descriptive nearest-rank 2.5/97.5 percentile ranges from all 3,125 ordered paired-block bootstrap resamples (ranks 79 and 3047), not coverage-guaranteed confidence intervals or significance tests. Goodput is 40 actual quality-qualified completion tokens divided by the sum of five client E2E durations per arm. Startup launch-to-ready probe times are separate and excluded from request latency. This single instrumented study supports no repetition-based confidence, power, or SOTA claim.}
  \label{f:paired-latency}
\end{figure*}
```

### read-wait

PDF: `papers/arxiv/figures/paired-read-read-wait.pdf`
PNG: `papers/arxiv/figures/paired-read-read-wait.png`

Read-demand wait and client-latency changes in the same five paired blocks. Each request wait is the sum of its anchored per-invocation host demand_wait_ns deltas, converted to seconds; each arm is the median of five measured-QA request waits. Connected points show blocking (blue circles) and lookahead (teal squares), with frozen execution order beside each block. The right panel plots lookahead-minus-blocking differences of the arm medians for request wait and client E2E; labels and circle/triangle markers retain order. All five blocks have lower observed wait and lower E2E with lookahead. This is an association consistent with less waiting, not proof that wait reduction caused the critical-path latency change. Host wait intervals are not CUDA timelines, are not additive to client latency, and do not establish GPU overlap. No fitted trend, causal coefficient, or extra independent replicates are inferred; shared-client/cache conditions remain a limitation.

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/paired-read-read-wait.pdf}
  \caption{Read-demand wait and client-latency changes in the same five paired blocks. Each request wait is the sum of its anchored per-invocation host demand\_wait\_ns deltas, converted to seconds; each arm is the median of five measured-QA request waits. Connected points show blocking (blue circles) and lookahead (teal squares), with frozen execution order beside each block. The right panel plots lookahead-minus-blocking differences of the arm medians for request wait and client E2E; labels and circle/triangle markers retain order. All five blocks have lower observed wait and lower E2E with lookahead. This is an association consistent with less waiting, not proof that wait reduction caused the critical-path latency change. Host wait intervals are not CUDA timelines, are not additive to client latency, and do not establish GPU overlap. No fitted trend, causal coefficient, or extra independent replicates are inferred; shared-client/cache conditions remain a limitation.}
  \label{f:paired-read-wait}
\end{figure*}
```

### accounting

PDF: `papers/arxiv/figures/paired-read-accounting.pdf`
PNG: `papers/arxiv/figures/paired-read-accounting.png`

Equal logical work and distinct memory/accounting scopes for all ten lifecycles. Each lifecycle records 128,811,263,616 logical checkpoint-reader bytes and 128,811,927,312 requested host-to-device bytes over its 12 requests: equal across modes, not equal to each other (H2D minus reader is 663,696 bytes). Every arm records 40,943 verified reads, zero wasted read bytes, and zero unknown read counts/bytes in the read-pipeline ledger. These application counters are not physical backend I/O or bus traffic. The owner-lifetime logical host-reservation maximum is 16,777,984 bytes (16.000732 MiB) against a 64 MiB logical limit; it is not RSS. Separately, the device-wide sampled maximum is 4,125 MiB against a configured 7 GiB target in every arm; it is neither an exact continuous peak nor exact process-owned GPU bytes. Bars are common observed maxima, never sums of lifetime peaks; do not add PSS, logical reservations, and device memory. Request-period shared NFS-client serverreadbytes are 314,825,320 for P1 blocking and recorded zero for the other nine arms. P1 blocking individual request windows sum to 312,732,264 bytes; the remaining 2,093,056 bytes lie between those windows within the enclosing shared-client phase. Fresh servers do not establish cold NFS/page caches, and shared-client counters do not identify physical NAS backend traffic. Startup/shutdown PID-I/O deltas remain unobserved (null), not zero. Zero final logical charges do not prove universal physical release. MiB and GiB denote 2^20 and 2^30 bytes.

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/paired-read-accounting.pdf}
  \caption{Equal logical work and distinct memory/accounting scopes for all ten lifecycles. Each lifecycle records 128,811,263,616 logical checkpoint-reader bytes and 128,811,927,312 requested host-to-device bytes over its 12 requests: equal across modes, not equal to each other (H2D minus reader is 663,696 bytes). Every arm records 40,943 verified reads, zero wasted read bytes, and zero unknown read counts/bytes in the read-pipeline ledger. These application counters are not physical backend I/O or bus traffic. The owner-lifetime logical host-reservation maximum is 16,777,984 bytes (16.000732 MiB) against a 64 MiB logical limit; it is not RSS. Separately, the device-wide sampled maximum is 4,125 MiB against a configured 7 GiB target in every arm; it is neither an exact continuous peak nor exact process-owned GPU bytes. Bars are common observed maxima, never sums of lifetime peaks; do not add PSS, logical reservations, and device memory. Request-period shared NFS-client serverreadbytes are 314,825,320 for P1 blocking and recorded zero for the other nine arms. P1 blocking individual request windows sum to 312,732,264 bytes; the remaining 2,093,056 bytes lie between those windows within the enclosing shared-client phase. Fresh servers do not establish cold NFS/page caches, and shared-client counters do not identify physical NAS backend traffic. Startup/shutdown PID-I/O deltas remain unobserved (null), not zero. Zero final logical charges do not prove universal physical release. MiB and GiB denote 2\textasciicircum{}20 and 2\textasciicircum{}30 bytes.}
  \label{f:paired-accounting}
\end{figure*}
```

## Estimator and denominators

- All five paired blocks qualify; no failed/censored/unissued slot is dropped. 120/120 requests completed across 10 fresh server lifecycles; 50 measured QA requests are nested within arms, not 50 independent replicates.
- E2E and TTFT: median of five measured requests per arm. Per-block ratio is L/B; aggregate is `exp(median(log(L/B)))`. Goodput: 40 actual qualified tokens divided by the sum of five client E2E durations, followed by the same block estimator.
- Exactly `5^5 = 3125` ordered resamples of five paired-block log-ratios; take each sample median, sort, take 1-based ranks 79 and 3047, exponentiate. These descriptive 2.5/97.5 percentile ranges are NOT a coverage-guaranteed 95% confidence interval. No significance, power, SOTA, or 50-request bootstrap.

| Outcome | Aggregate L/B | Descriptive range |
|---|---:|---:|
| E2E | 0.91823466 | [0.89303465, 0.94946064] |
| TTFT | 0.88755304 | [0.86003892, 0.94888011] |
| Qualified goodput | 1.10908224 | [1.07392849, 1.13497822] |

## Reading the plots without overclaiming

- Palette: Blocking blue `#0072B2`, Lookahead teal `#008577`; circle/square endpoints, direct mode/order labels, and hatching retain meaning in grayscale. Ratio and change points encode order by circle (B/L) versus triangle (L/B), not mode. Neutral diamonds denote aggregate estimates, and neutral gray strips denote descriptive ranges.
- Read-wait is the per-request sum of host pipeline deltas, then a within-arm median. The scatter compares differences of medians, not the median of requestwise differences and not a causal decomposition. E2E is send-to-EOF; TTFT is send-to-first nonempty SSE content. Neither is reconstructed from worker token counts.
- Equal work is checked for each arm against the 102 actual invocation records and 12 request windows. No route-ID signature or cache-miss equivalence is inferred. Source bytes and H2D differ by 663,696 bytes per lifecycle; neither counts physical storage traffic.
- Host logical reservation and device samples use separate units/axes and scope labels. Lifetime peak fields are maxima, not additive deltas. PSS is deliberately not plotted or added. Final zero logical charges are not physical-release proof.
- Shared NFS request-phase `serverreadbytes`: P1 blocking 314,825,320 bytes; others 0. The P1 blocking individual request windows sum to 312,732,264 bytes; 2,093,056 bytes lie between those windows inside the enclosing phase and are not assigned to requests. Startup and shutdown NFS phase windows are separately checked. Missing startup/shutdown PID-I/O and startup logical-reader accounting stay null. Physical backend bytes, owned GPU attribution, continuous device peak, and causal critical-path savings stay null.
- Actual quality coverage: 900 worker-enqueued token IDs, joined through worker/HTTP IDs and ordinals; 110 exact eight-token oracle requests plus 10 two-token smoke requests validated for usage/length only. The 1,020 invocation rows are not tokens or replicates. SSE content events are not token boundaries.
- The two earlier stopped studies remain excluded, with their original denominators retained in plot data. The public checker validates their archived projections without importing private receipts. The actual source archive count is 1,484, checked against the public source/build projection rather than copied from an older analysis.

## Separate startup probe observations

These are recorded launch-to-ready probe durations, **not** a subtraction from the coordinator/lifecycle start timestamp, not request latency, and not repeat-study statistics.

| Arm | Startup probe (s) | Device samples |
|---|---:|---:|
| P1-blocking | 12.849807 | 54 |
| P1-lookahead | 13.151951 | 50 |
| P2-blocking | 13.049569 | 53 |
| P2-lookahead | 12.847339 | 49 |
| P3-lookahead | 13.148421 | 50 |
| P3-blocking | 12.948767 | 53 |
| P4-lookahead | 12.967943 | 50 |
| P4-blocking | 13.246785 | 53 |
| P5-blocking | 13.050234 | 53 |
| P5-lookahead | 13.051107 | 50 |

## Public inputs, provenance, and deterministic reproduction

The renderer calls the existing `publication_results_v1/check.py:verify()` before
plotting. This verifies **all 623 public files**, including compressed and expanded
member hashes, original source pointers, token/HTTP joins, invocation/counter
reconciliation, and all 3,125 bootstrap draws. The 622 manifest members total
44,142,418 expanded bytes; the complete archive is 4,451,307 bytes. The original
raw-file inventory count is 1,461, not a claim that private files are required.
No cached `plot-data.json`, private `inputs.json`, `derive.py`, `receipts.py`, or
`study.py` is imported by the public rendering path.

Public collector receipts retain sampled device maxima and sample counts, not
the raw device-sample stream. Regeneration checks their mutually consistent
summary fields and preserves their sampled, device-wide scope; it does not
invent samples, continuous peaks, process ownership, or missing physical I/O.
Waits are recomputed from the checker's invocation-derived request sums, and the
host gauge is a maximum, never a sum. The NFS phase/window gap remains 2,093,056
bytes for P1 blocking. The label **Requested host-to-device bytes** is the existing
`h2d_bytes` application counter, exactly **128,811,927,312** per lifecycle—not
measured physical bus traffic.

From the repository root, using Python with the recorded dependencies:

```sh
python3 -B papers/arxiv/experiments/storage_demand/paired_read_v1/analysis_v1/render_public.py --output-dir /tmp/paired-read-public
```

Add `--check` to that same command to verify actual output hashes, reconstruct the
public dataset again, re-render, and compare every byte without rewriting outputs.
`plot_paired_read.py` delegates to this public path by default, including `--check`.
The output directory is mandatory and must be outside the frozen public archive.
All ten output basenames are portable: three `paired-read-*.pdf`, three matching
PNGs, `paired-read-preview.png`, `plot-data.json`, `layout.json`, and
`plot-manifest.json`. The repository keeps the seven images in `papers/arxiv/figures/`
and the three JSON outputs here; compare those files to the identically named
files in the generated directory. The CLI never updates those locations implicitly.

`plot-manifest.json` binds the actual generated bytes, checked public-manifest
hash, generator/checker source hashes, and renderer/font pins. Its only recursive
omission is its own hash; `--check` still compares its regenerated bytes. PDF/PNG
metadata embeds public input-manifest identity, generator, renderer, toolchain,
and plot-data hashes. Dates, host paths and random identifiers are absent.
The public archive's frozen `manifest.json` remains verbatim; no resealing is
needed. New provenance metadata intentionally changes image file hashes, not the
plotted measurements. The only visible change from the earlier figures is the
accounting label and the spacing needed for its bounded two-line layout.

### Renderer environment and detached export

`requirements-render.txt` records installed package versions; `renderer-pins.json`
checks Python 3.12.4, Matplotlib 3.8.4, NumPy 1.26.4, Pillow 10.3.0, FreeType 2.12.1,
zlib 1.2.13, transitive rendering dependencies, and the bundled DejaVu Sans normal
and bold font hashes. The renderer explicitly selects those font files rather
than accepting a host-font substitution. It installs nothing. Package version
pins alone do not guarantee the same linked FreeType/zlib build; byte-identical
reproduction is scoped to the full recorded environment, and mismatches fail
before rendering. The already available validated executable in the recording
environment is `/home/admin/wxw403883/anaconda3/bin/python3.12`; the default
Python 3.10.9 there lacks Matplotlib.

For a detached export, copy **the entire unmodified `publication_results_v1/`**
next to an `analysis_v1/` containing only:

- `render_public.py`, `render.py`, `plot_common.py`;
- `renderer-pins.json`, `requirements-render.txt`;
- optionally `plot_paired_read.py` for the legacy command name, and
  `tests/test_public_render.py` for the public regression suite.

The pinned Matplotlib installation supplies the two hashed font files; no private
source or original experiment tree is necessary. In that export, run
`python3 -B analysis_v1/render_public.py --output-dir /tmp/paired-read-public`.

Run the public-only tests in the pinned rendering environment:

```sh
python3 -B -m unittest discover -s papers/arxiv/experiments/storage_demand/paired_read_v1/analysis_v1/tests -p test_public_render.py -v
```

Tests construct detached exports **with the private tree absent**, independently
recompute arm/request values and intervals from public observations, check NFS
windows and logical gauges, verify PNG/PDF metadata and actual artifact hashes,
and demand byte-identical local/detached outputs. Tampered inputs, output bytes,
and substituted font hashes fail closed. `layout.json` records visible text
bounds and rejects out-of-canvas or overlapping text, including the wrapped
accounting label; it does not prove absence of marker/line collisions.

### Deprecated custodian-only path

Custodians retaining the original 840 private inputs may still run
`plot_paired_read.py --private-inputs --output-dir /tmp/paired-read-private`
(and add `--check`). This separate legacy path is **not public regeneration** and
has different provenance bytes. Its private-input tests are
`python3 -S -B -m unittest discover -s .../analysis_v1/tests -p test_paired_plots.py -v`.
Neither command is needed to regenerate the publication figures.

## Exact plot text

The authoritative complete, generated text inventory is in `layout.json` (titles, panel labels, tick labels, values, legends, notes). The captions above describe the unchanged measurements and estimator; this guide is maintained separately and does not insert captions into the manuscript. Figures are 7.5 inches wide and intended for approximately full two-column text width. `paired-read-preview.png` is the vertical contact sheet.
