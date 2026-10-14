# Fixed storage-demand receipt analysis

A CPU-only, standard-library summarizer for the two frozen TP1 middle attempts
`storage-demand-tp1-middle-20261013-01` and `-02`. It does not import the harness,
launch a server, query GPUs, read checkpoint payloads, or modify input receipts.

`inputs.json` seals the 41 consumed files using repository-relative paths and
SHA-256. Generation verifies both this seal and the existing report receipt
pins. Missing counters remain null; malformed schemas or discrepancies fail
closed without replacing existing outputs. There is no automatic resealing.

From the repository root:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/analysis/tests -v
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/analysis/summarize.py
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/analysis/summarize.py --check
```

Summarizer outputs are restricted to `output/summary.json`, `output/TABLES.md`, and
`output/REPORT.md`. Check mode recomputes these bytes without writing and rejects
stale summaries, tables, or reports. Output is deterministic: no fresh timestamp,
machine identity, or mtime is included. Read `output/REPORT.md` for counter
semantics and limitations before quoting the descriptive tables.

## Colored experimental plots

`plot_storage_demand.py` independently verifies the sealed sources and the
existing summarizer outputs before deriving plot values. It never reseals inputs
or writes the original summary artifacts. Rendering uses the already installed
Matplotlib/Agg toolchain; the data derivation and dedicated tests are stdlib-only.

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -B papers/arxiv/experiments/storage_demand/analysis/plot_storage_demand.py
/home/admin/wxw403883/anaconda3/bin/python3.12 -B papers/arxiv/experiments/storage_demand/analysis/plot_storage_demand.py --check
```

The four PDF/PNG figure pairs and contact sheet use only new
`papers/arxiv/figures/storage-demand-*` paths. `output/plot-data.json` binds inputs,
summary and generators; `output/plot-manifest.json` hashes every figure and records
renderer/font versions. Check mode re-renders all expected bytes without replacing
artifacts; cross-version renderer byte equivalence is not assumed.

See [`output/PLOTS.md`](output/PLOTS.md) for exact paths, complete captions,
paper-ready LaTeX inclusion snippets, equations, configured ceilings and unknown
quantities. These figures show logical demand, entry churn/reconciliation,
separate cache gauges, and single-DEBUG-run timing—not physical I/O, cache misses,
full numerical acceptance, a physical-capacity proof or comparative speedups.
The recorded `cache.peak_bytes` is cache charged-byte high-water (resident + pending
+ scratch), bounded by `ExpertCacheLimits.max_bytes` = 320 MiB, a cache accounting
ceiling, not owner physical memory. An owner-physical peak or ceiling is not
independently recorded and is not fabricated from the cache accounting budget. The initial zero-entry occupancy is implied by the
first counter balance, not an independent pre-request sample.
