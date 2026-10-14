# Storage-demand receipt plots

Single successful TP1 DEBUG run (-02), n=1; -01 failed before server launch. No physical-I/O, cache-miss, numerical-acceptance, or speedup inference.

Hello2: HTTP/usage only (reference null). Three canaries: text + eight-output-usage checks only. Generated IDs, exact-token acceptance and numeric acceptance: unknown.

## Figures and exact paper includes

Use the following snippets from `papers/arxiv` with the existing `graphicx` package. They are suggestions only; the generator does not edit the manuscript.

### logical-transfer

PDF: `papers/arxiv/figures/storage-demand-logical-transfer.pdf`
Preview: `papers/arxiv/figures/storage-demand-logical-transfer.png`

Paired logical checkpoint-reader and facade H2D byte deltas for four serialized requests in storage-demand TP1 middle -02. GiB means 2^30 bytes; the right panel resolves the small H2D-minus-reader differences in KiB (2^10 bytes). These are application counters, not physical storage I/O or physical bus traffic; startup I/O is outside invocation accounting.

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/storage-demand-logical-transfer.pdf}
  \caption{Paired logical checkpoint-reader and facade H2D byte deltas for four serialized requests in storage-demand TP1 middle -02. GiB means \(2^{30}\) bytes; the right panel resolves the small H2D-minus-reader differences in KiB (\(2^{10}\) bytes). These are application counters, not physical storage I/O or physical bus traffic; startup I/O is outside invocation accounting.}
  \label{fig:storage-demand-logical-transfer}
\end{figure*}
```

### cache-reconciliation

PDF: `papers/arxiv/figures/storage-demand-cache-reconciliation.pdf`
Preview: `papers/arxiv/figures/storage-demand-cache-reconciliation.png`

Expert entry reconciliation and separate cache gauges in storage-demand TP1 middle -02. The 11,519 uploads minus 11,455 evictions equal 64 entries, matching the last observed resident-entry gauge; the implied initial occupancy is zero, not independently measured. Subsequent observed boundaries satisfy the entry balance. The recorded cache.peak_bytes value, 209.574 MiB, is the cache charged-byte high-water (resident + pending + scratch), bounded by ExpertCacheLimits.max_bytes = 320 MiB, a cache accounting ceiling, not owner physical memory. No independently recorded owner physical peak or owner physical ceiling is available; no contradiction is implied. Invocation gauges and lifetime peaks are not accumulated across invocations or a final owner ledger.

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/storage-demand-cache-reconciliation.pdf}
  \caption{Expert entry reconciliation and separate cache gauges in storage-demand TP1 middle -02. The 11,519 uploads minus 11,455 evictions equal 64 entries, matching the last observed resident-entry gauge; the implied initial occupancy is zero, not independently measured. Subsequent observed boundaries satisfy the entry balance. The recorded \texttt{cache.peak\_bytes} value, 209.574 MiB, is the cache charged-byte high-water (resident + pending + scratch), bounded by \texttt{ExpertCacheLimits.max\_bytes} = 320 MiB, a cache accounting ceiling, not owner physical memory. No independently recorded owner physical peak or owner physical ceiling is available; no contradiction is implied. Invocation gauges and lifetime peaks are not accumulated across invocations or a final owner ledger.}
  \label{fig:storage-demand-cache-reconciliation}
\end{figure*}
```

### timing

PDF: `papers/arxiv/figures/storage-demand-timing.pdf`
Preview: `papers/arxiv/figures/storage-demand-timing.png`

Launch-to-ready startup and per-request HTTP TTFT/E2E in the single successful DEBUG-instrumented storage-demand TP1 middle -02 run (n=1). TTFT is send to first nonempty SSE content; E2E is send to EOF. Startup has a different origin and is not part of the request bars. Instrumentation overhead is unmeasured; there are no statistics, confidence intervals, kernel timings, or comparative speedups.

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/storage-demand-timing.pdf}
  \caption{Launch-to-ready startup and per-request HTTP TTFT/E2E in the single successful DEBUG-instrumented storage-demand TP1 middle -02 run (n=1). TTFT is send to first nonempty SSE content; E2E is send to EOF. Startup has a different origin and is not part of the request bars. Instrumentation overhead is unmeasured; there are no statistics, confidence intervals, kernel timings, or comparative speedups.}
  \label{fig:storage-demand-timing}
\end{figure*}
```

### invocation-timeline

PDF: `papers/arxiv/figures/storage-demand-invocation-timeline.pdf`
Preview: `papers/arxiv/figures/storage-demand-invocation-timeline.png`

Per-invocation expert upload/eviction deltas and resident-entry observations for the 30 completed invocations in storage-demand TP1 middle -02. Serialized request/log-byte anchors assign the 3/9/9/9 invocations to requests. The x-axis is log-order invocation ordinal, not elapsed time, output-token index, or transaction-as-request ID. Entry occupancy is observed only at invocation completion, not continuously; uploads are not inferred cache misses.

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/storage-demand-invocation-timeline.pdf}
  \caption{Per-invocation expert upload/eviction deltas and resident-entry observations for the 30 completed invocations in storage-demand TP1 middle -02. Serialized request/log-byte anchors assign the 3/9/9/9 invocations to requests. The x-axis is log-order invocation ordinal, not elapsed time, output-token index, or transaction-as-request ID. Entry occupancy is observed only at invocation completion, not continuously; uploads are not inferred cache misses.}
  \label{fig:storage-demand-invocation-timeline}
\end{figure*}
```

## Mathematics and limits

- GiB = 2^30 bytes; MiB = 2^20 bytes; KiB = 2^10 bytes. Seconds come only from bound HTTP/launch receipts.
- Per-request reader/H2D bytes sum only additive invocation deltas. The KiB inset prevents rounded GiB labels from implying exact equality.
- Entry reconciliation is count arithmetic, not byte conservation. Evicted-byte deltas are not recorded, so no byte-flow balance is fabricated.
- Initial occupancy is inferred as zero by the first balance equation, not an independently observed pre-request gauge. Every later observed boundary reconciles exactly.
- Entries: 64; cache accounting ceiling: 335,544,320 bytes = 320 MiB; scratch cap: 8,388,608 bytes = 8 MiB. Resident bytes are one component of the cache charge, not a separate resident-only byte budget. The bound startup log confirms the binary meaning of the legacy scratch-MB flag.
- `cache.peak_bytes` (raw receipt key `expert_owner_peak_bytes`) is cache charged-byte high-water (resident + pending + scratch), bounded by `ExpertCacheLimits.max_bytes` = 320 MiB, a cache accounting ceiling, not owner physical memory.
- The 209.574 MiB value is a cache accounting peak, not an independently recorded owner-physical peak. Owner physical peak/ceiling remain unobserved; no contradiction with the 320 MiB cache accounting ceiling is implied. Pending-upload/quarantine observations are zero but do not prove shutdown closure.
- Device budget (7168 MiB) and workspace (2048 MiB) have different scopes; they are not cache ceilings and are not summed with cache gauges.
- These are configured accounting limits, not physical capacity. The manuscript admission and credit-conservation equations lack full ledger observations here; the figures do not purport to verify them.
- The timeline is log-order only, mapped by actual byte anchors, with no interpolated samples or fabricated token/time alignment.

## Reproduction and provenance

Run from the repository root using the already installed Python 3.12 / Matplotlib toolchain:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/analysis/summarize.py --check
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/analysis/tests -v
/home/admin/wxw403883/anaconda3/bin/python3.12 -B papers/arxiv/experiments/storage_demand/analysis/plot_storage_demand.py
/home/admin/wxw403883/anaconda3/bin/python3.12 -B papers/arxiv/experiments/storage_demand/analysis/plot_storage_demand.py --check
```

The data derivation and tests use only stdlib (`-S` supported); rendering lazily uses existing Matplotlib/Agg, NumPy and Pillow. No installs, network, GPU calls, harness imports or TeX engine are needed.

`plot-data.json` binds all 41 sealed receipts, the checked summary, input manifest, summarizer and plot-generator hashes. `plot-manifest.json` seals data, guide, every PDF/PNG and the contact sheet, and records renderer/font versions. PDFs and PNGs embed the input/generator/data hashes. No current date, hostname, random identifier or mtime is included. `--check` re-renders in memory and compares every expected byte without overwriting artifacts or resealing inputs. Byte reproducibility is scoped to the recorded renderer/font toolchain, not promised across arbitrary Matplotlib/FreeType versions.

Preview contact sheet: `papers/arxiv/figures/storage-demand-preview.png`.

Unknown quantities remain null and are excluded from plotted series. -01 is documented as unexecuted, not inserted as a zero-valued observation. No error bars, fitted model, distribution summary, or statistical denominator is manufactured.
