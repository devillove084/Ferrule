# CPU publication handoff — 2026-10-15

## Delivered scope

Only new files under `publication_pressure_v1/` and new
`papers/arxiv/figures/cache-pressure-*` files were written. No production,
protocol, original raw, manuscript, prior figure, or other publication files
were changed. No GPU query/lock, model execution, server, HTTP request,
component fault, profiler, Git staging, commit, or branch creation occurred.
The `/papers/` tree is ignored; tracked Git status remained clean.

The completed study remains `artifacts-go-v2-20261015-02`: **3 lifecycles,
12 successful requests, 9 exact eight-ID canaries, 78 tokens**. This is n=1,
fixed-order, jointly varied entries/bytes, blocking only. No causal speedup,
independent LRU effect, physical frontier, continuous peak, physical NAS
traffic, or universal physical release claim is introduced. Earlier no-run,
Chinese-censored, and bound-pin-start-rejected states are kept separate.

## Checks actually run

- `python3 -B .../publication_pressure_v1/curate.py`: PASS. Rehashed every
  original artifact against the unchanged **244-file** source manifest.
- `python3 -B -m unittest discover -s .../publication_pressure_v1 -p 'test_*.py' -v`:
  **14 tests passed**. Includes altered actual token IDs, missing HTTP usage
  despite a full worker trace, missing DONE, falsely assigning an exact-ID
  oracle to Hello2, extra admissions without tokens, changed/missing counters,
  summed owner peaks, missing memory samples, unpaired helper spans, fabricated
  physical admission/release, and separation of historical failure outcomes.
- `python3 -B .../publication_pressure_v1/verify.py --check-source .../artifacts-go-v2-20261015-02`:
  PASS. Independently derived public scalars equal the sealed scalar projections;
  all 244 originals and five supplemental plan/history inputs still match.
- Deterministic recuration: PASS. `records.jsonl.gz`, `summary.json`,
  `history.json`, and `manifest.json` were byte-identical on repeat extraction.
- Python AST parsing: PASS for all new publication and figure Python sources.
- Public-data coordinate scan: PASS for private home paths, NAS mount paths,
  GPU UUIDs, loopback endpoint addresses, and authorization/Bearer patterns.
  This is a scoped scan, not a general secret-detection certification.
- `papers/arxiv/figures/cache-pressure-budget.py`, invoked with the preinstalled
  Python 3.12.4 interpreter and `-B`:
  PASS with the already-installed Matplotlib **3.8.4**, Agg backend. PDF and PNG
  generated from checked raw public records. No package installation was needed.
- Figure preview inspected: four colored panels; readable three-line point
  labels, separated GPU caveat/value labels, zero-based axes, dashed requested
  ceiling and non-stacked charge/peak bars. The PDF header/end marker and PNG
  dimensions (**2100 × 1560**) were also checked.

The extraction/check/test runtime was Python **3.10.9**. The rendering runtime
was Python **3.12.4**. No Cargo test/build or GPU benchmark was necessary for
this isolated publication-only change.

## Public data inventory

- **1,516 typed records**, 1,472,524 uncompressed bytes.
- **115,866 compressed bytes** in deterministic `records.jsonl.gz`.
- All **90** invocation summaries retain their fields and typed nested ledgers.
- **78** actual worker token enqueues; **12** complete HTTP/SSE observations.
- **108** memory samples and **42** shared-client NFS snapshots.
- **1,074** timing records, preserving **537** completed named wall/thread-CPU
  spans. Nested and concurrent totals are not added across phases.
- The compact size comes from retaining primitive evidence once instead of
  copying repeated nested lifecycle/assessment trees; no invocation field,
  request, timing event, memory sample, or token enqueue was sampled away.

## Content hashes

| Artifact | SHA256 |
|---|---|
| Original 244-file manifest / unchanged `source-manifest.json` | `c570b05aa5c6ffc63c588c7e78441728cf1c1db9119e83a1a9a3c813efb7a283` |
| `manifest.json` | `a614c503035cfcde8564ca825e571541e925c1d228c4014fe0fd9d7f605b9164` |
| `records.jsonl.gz` | `db377629e2a48c7681b022ee4e13f228b75e9be6a7c11cb81693c037272fcb43` |
| `summary.json` | `5bbba89b64d938e966e47dfe53dd5c10dd5110fb2186e2f43a444b5645c2a947` |
| `cache-pressure-budget.py` | `9c7f282db04ef3e56dd8755aa2117002c2ca2c9e2c7d24eae7ec9ac45504a36d` |
| `cache-pressure-budget.pdf` | `dfd82ce79fa698678546ed38c4bdcfc96cd16167395f8b92136122efe6770911` |
| `cache-pressure-budget.png` | `8d6fea8e53bbe2a7ed428beb13689b37e31c6eed2048b2337beb201852eadc61` |

These hashes bind the local handoff, not an external attestation. Public
projections do not replace original raw receipts or prove physical attribution.
The original REPORT/final result/final handoff remain untouched and are covered
by the 244-file rehash. Commands are listed with full repository-relative paths
in `README.md`.
