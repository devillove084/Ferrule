# Scoped read–compute overlap: portable actual-05 evidence

This is a **curated public projection of two fresh actual-05 captures**, not a
new collection, a full Nsight export, or an upgrade of any old capture. It is
CPU-only to check and plot. No GPU, CUDA/Nsight SDK, network, model, checkpoint,
private directory, or production-server launch is needed for public checks.

**Both arms are accepted only for the declared overlap scope. Full-trace
acceptance remains false. Dropped-event count remains unknown (`null`).**
The three official wrapper warnings per arm are retained, not silently ignored.
There is no ordinary-performance confidence interval or physical NAS-service
claim in this archive.

## Reproduce offline

From this directory, using Python 3.9+ and only its standard library:

```sh
python3 -S -B check.py
python3 -S -B -m unittest discover -s . -p test_publication.py -v
```

From the repository root, render with the existing Matplotlib toolchain:

```sh
python3 -B papers/arxiv/figures/read-compute-overlap.py
python3 -B papers/arxiv/figures/read-compute-overlap.py --check
python3 -B -m unittest discover -s papers/arxiv/experiments/storage_demand/profile_overlap_v1/publication_v1 -p test_figure.py -v
```

The local validated renderer uses Python 3.12 and Matplotlib 3.8.4. Exact version,
FreeType, NumPy, Pillow, zlib, and font-file SHA256 records are in `figure.json`.
`--check` compares PDF, PNG, and figure metadata **byte for byte**, so it requires
the recorded renderer. Other compatible Matplotlib installations may render
correctly but not produce identical bytes. The arithmetic checker has no such
third-party requirement. To keep a different render separate, supply
`--output-directory` and a copy of the archive via `--archive`; rendering writes
`figure.json` in the chosen archive. Copy this directory and the plot source to
another machine, then use `--archive PATH`, for repository-independent plotting.

`export_private.py --check` is a separate maintainer-only, CPU-only re-extraction
from the existing private inputs. It opens SQLite with `mode=ro`, checks source
and final-handoff pins, and compares every public evidence file without writing.
It does **not** invoke the original collection workflow or transfer acceptance.
Its SQL string constants are the exact extraction queries. Running without
`--check` creates the evidence files only when they do not already exist.

## Exact whole-request measurements

Intervals are half-open integer-nanosecond ranges on the common Nsight clock,
translated so each QA request-window start is zero. There is no clock alignment
with Rust `Instant`, no absolute UTC, and no synthetic padding of GPU activity.

Let `U(X)` merge the intervals of `X`, including identical/overlapping records,
and let `|U(X)|` be their total length. Let `R` be worker `read_subset` envelopes,
`P` be contained OSRT `pread`/`pread64` wall intervals, and `K` be actual GPU
kernel intervals whose uniquely correlated launch APIs are contained in
`layer.expert` (and not `layer.expert.upload`). Report:

- Envelope overlap: `|U(R) ∩ U(K)|`, divided by `|U(R)|` for percent.
- File-call overlap: `|U(P) ∩ U(K)|`, divided by `|U(P)|` for percent.
- GPU-active duration: `|U(K)|`, **not** first-to-last kernel elapsed time,
  a CPU submission range, memcpy, upload, or time before the first kernel.

| Measurement | Blocking | Lookahead |
|---|---:|---:|
| Worker read-envelope union (s) | 2.982515538 | 2.989969321 |
| OSRT pread-wall union (s) | 2.299019830 | 2.295077102 |
| Actual expert-kernel union (s) | 0.194741187 | 0.194217266 |
| Read-envelope ∩ kernel union (s) | 0 | 0.174409210 |
| Envelope overlap / envelope union | 0% | 5.833143797665073% |
| Pread-wall ∩ kernel union (s) | 0 | 0.000751507 |
| Pread overlap / pread union | 0% | 0.032744302984205366% |

Each arm contains one QA, nine decoder invocations, 3,678 worker read ranges,
3,678 expert submission ranges, 3,678 upload ranges, and 22,068 contained preads
(exactly six per read range). Each has 33,403 attributed expert kernels out of
63,042 kernels in the original whole export. Both QAs pass with the same eight
generated IDs: `[25358, 13, 198, 760, 6511, 314, 6124, 369]` (5 prompt tokens,
8 completion tokens). Whole-export observed launch/kernel closure has no orphan
or duplicate correlation keys; that is not a proof that simultaneous missing
API and activity records are impossible.

### What the byte count means

Each arm has **11,571,399,936 source-verified requested/packed bytes**. This is
bound to the frozen source and the complete typed owner-lifetime read ledger:
submitted, worker-started, I/O-started, completed, and verified counts agree;
failed/cancelled/unknown/unobserved counts are zero; requested bytes sum to
`proved_read_bytes` and shared-reader interval bytes.

It is **not an OSRT byte total**. The actual OSRT schema has no size field or
proved `ssize_t` encoding. All observed pread `returnValue`s are zero and remain
opaque. Every public pread row carries `osrt_bytes=null`; aggregate
`osrt_read_bytes` is also null. Zero does not prove a zero-byte read or individual
call success. Source verification is whole-operation evidence, not a complete
per-syscall success journal. Shared-reader counters are not ticket-certified.
The last receipt is one positive identity check, not a complete read-operation
to layer/expert mapping. We do not invent that mapping in the CSVs.

The frozen compiled CLI is from commit
`86c58a017d7c4b53490f536a3b1a133f2f7848cc`, SHA256
`7cc8e0ff6e84f27090494b8e9e27dc3a388d1d2f3fc34ae39dae416e1e607a01`.
The measurement correction is commit
`5e4e29cb883c84392dfaff15120ee910dac093ad`. The source archive SHA256 and eight
frozen source-member hashes are in `source.json`, alongside short source
excerpts showing the actual `FileExt::read_at` loop, allocation, NVTX boundary,
and the **subsequent** `SourceVerified` boundary. Source and image generations
are both 1; the generation is checked against the current owner ledger.

The NVTX read scope covers allocation and the actual source read call. It
excludes session opening and pre/post source verification. OSRT duration is
userspace file-API wall time, which can include CPU interrupts/preemption and
page-cache service. Neither track establishes active device I/O, physical NAS
service time/bytes, CPU scheduling state, or critical-path causality. The many
short expert kernels and intervening gaps are retained rather than converted
to one broad compute envelope. These are profiled QA executions; their E2E
latencies must not be pooled with ordinary timing pairs or treated as a speed
confidence result.

## Public files and pruning

The six CSV gzip files occupy **2,924,573 bytes** together. Evidence payloads
including JSON and the evidence manifest occupy **2,987,128 bytes**, before
code, documentation, and the figure record. CSVs contain 133,010 data rows:
66,505 per arm, with **one row per actual source event**, not copies of both
per-range unions and their underlying events. Gzip has no filename or timestamp.
The hash/size manifest covers the complete evidence payload set.

| File | Public content |
|---|---|
| `manifest.json` | Exact evidence file set, compressed sizes, SHA256s |
| `source.json` | Frozen build identity, source excerpts/hashes, read measurement contract, final handoff projection, redaction policy |
| `history.json` | Original collection outcomes and separately labeled retrospective -04 analysis |
| `*-nvtx.csv.gz` | All 11,034 closed read/expert/upload ranges per arm, original numeric metadata and SQL row IDs |
| `*-pread.csv.gz` | 22,068 request-scoped OSRT calls per arm, original API identities/return encodings, validated range joins, null byte field |
| `*-kernel.csv.gz` | 33,403 actual expert kernels per arm, original correlation IDs, launch API fields/intervals, validated expert-range joins |
| `*-proof.json` | Preliminary and final scope states, all diagnostic rows (projected), collector headers/schema, roles, counts, ledger, QA, maps summary, original hashes |
| `check.py` | Standalone integrity/semantic checker; recomputes interval unions, intersections, coverage, and attribution |
| `export_private.py` | Maintainer-only read-only SQL projection and source-pin reconciliation |
| `test_publication.py`, `test_figure.py` | Corruption, scope, arithmetic, empty-render, and layout regressions |
| `figure.json` | Derived full-request numbers, selected zooms, renderer/fonts, layout bounds, PDF/PNG hashes |

CSV column order is defined and enforced in `check.py`. Optional fields use the
literal `null`, never an ambiguous empty cell. `global_pid` is the original
serialized process key; `global_tid` retains the thread suffix. PID extraction
is `(global_pid >> 24) & ((1 << 24) - 1)`, and process identity from a thread is
`global_tid & ~((1 << 24) - 1)`. IDs are capture-local, not cross-arm identities.
`row_id` identifies an actual source-table row. In kernel CSVs, `api_row_id`
and `api_table` identify the correlated launch row. Kernel name IDs are retained
as numeric StringIds references; private symbol dictionaries/callchains are not
needed for timing or joins. `source_generation`/`operation` exist only on read
ranges, while image/layer/expert/rows exist only on expert/upload ranges.

The exporter first checks **all** observed math-launch/kernel correlation keys,
then retains only expert kernels. Non-expert and outside-request kernel counts
are separately recorded. It independently counts every OSRT process/event
class, retaining start/finish lifecycle counts separately from API counts, then
retains only target-request preads. Excluded startup preads remain in the count
proof. Unrelated runtime, OSRT, scheduling, memcpy and other SQL rows are not
copied into the public archive. The exporter cross-checks every retained read's
pread intervals and every expert's kernel union against the stored auditor,
not merely the aggregate overlap percentages.

### Redaction and trust boundary

This is an **explicit positive field projection**, not a recursive scrub of a
raw database. Credentials, nonces, worker authorization, environment, command
arguments, application maps, private logs, filesystem/host paths, physical GPU
UUIDs, absolute UTC/capture origins, and unrelated SQL payloads are omitted.
No `.sqlite`, `.nsys-rep`, `.nsyscapture`, shared object, SDK, raw manifest, or
application-map dump is distributed.

All 20 diagnostics per arm are represented with original severity, source,
process key, timestamp type, and classification. Diagnostic timestamps are
omitted because their mixed clocks are not silently relabeled as request time.
Two informational path-bearing texts per arm are explicitly redacted, with
original-text and canonical-original-row SHA256s; the three warnings retain
exact text. The raw collector schema and selected capture/export header fields
are preserved, including full/unfiltered export settings and OSRT threshold.

Original report, SQLite, audit, final analysis, collector receipt, roles,
quality, read ledger, maps, worker/custody/supervisor, export receipt and log
hashes/sizes are published. Authenticated roles are **projections of the sealed
private audit**, not independently reauthenticated by the public checker.
Map summaries retain ready/after checks (16 required, 19 observed) and raw-map
hashes, but not maps themselves. Backing-file/source hashes do not measure live
mapped-memory contents. Physical final release remains unobserved.

SHA256 pins allow a holder of the originals to check equality; they are not
signatures and cannot prove the content of withheld evidence to a reader who
has only the projection. The public checker verifies internal consistency and
reproducible arithmetic. It does **not** claim to replay omitted private
lifecycle authentication, prove universal trace completeness, or exclude
simultaneous unreported loss. `private_lifecycle_replayed=false` is explicit in
its output.

## Warning scope and preserved history

Every arm retains these exact warnings on its authenticated non-CUDA wrapper:

1. `No NVTX events collected. Does the process use NVTX?`
2. `CUDA profiling might have not been started correctly.`
3. `No CUDA events collected. Does the process use CUDA?`

The whole-export proof reports zero wrapper CUDA API, kernel, and NVTX payload
rows. No target warning or known loss diagnostic is present. The checker
rejects an unknown warning, a known warning moved to the model/global process,
a removed warning/header/role/source field, or changing the unknown drop count
to zero. These three wrapper warnings remain full-trace failures. The original
preliminary SQL audit remains `status=rejected`, with scoped SQL consistency
true and scoped acceptance false. Only the final scoped result has all seven
outer lifecycle gates and scoped acceptance true; its full-trace status is
still false.

| Generation | New captures | Original outcome | Accepted scoped arms | Full trace |
|---|---:|---|---:|---|
| -01 | 1 | Failed/rejected | 0 | false |
| -02 | 0 | Blocked before GPU | 0 | false |
| -03 | 0 | Blocked before GPU; direct custody adapter unverified | 0 | false |
| -04 | 1 | Rejected; lookahead unissued | 0 | false |
| -04 retrospective v2 | 0 | Corrected predicate consistent; **not historical acceptance** | 0 | false |
| -05 | 2 | Two new, prospectively scoped captures accepted | 2 | false |

The -04 original rejection was the old interpretation of OSRT zero as an
empty/failed read and resulting byte/count mismatch. Its original handoff and
zero accepted-arm count are unchanged. The new retrospective analysis reports
a corrected predicate but explicitly sets historical transfer/upgrade false.
It contributes no new capture or accepted arm. The history therefore records
four collected captures in total and only two accepted scoped arms, all from
-05; the rejected captures have not been erased from the denominator.

## Figure and caption

Source: `papers/arxiv/figures/read-compute-overlap.py`.
Outputs: `papers/arxiv/figures/read-compute-overlap.pdf` and `.png`.

The two three-track zooms use a rule fixed before rendering: take the first
source-generation-1 `read_subset` after the request bound and show the next
20 ms. This selects operation 1 in both arms, at request-relative offsets
7.435922 ms (blocking) and 8.662259 ms (lookahead). No selection uses the maximum
positive overlap. All events intersecting those windows are shown (6/10 read
ranges, 36/60 preads, 46/82 kernels); this is a deterministic excerpt, not a
random sample or an estimate of full-request overlap. Kernel gaps and leading
delay are not filled. Small start glyphs improve visibility without changing
interval widths. Operation labels sit above, not on, the intervals. The two
aggregate bars have different, explicitly labeled millisecond scales.

**Suggested caption:** *Scoped overlap for two fresh profiled QA executions of
the same serving executable. Top: deterministic 20-ms excerpts beginning at each arm's
first source-qualified worker read, showing allocation-plus-read NVTX
envelopes, OSRT pread wall intervals, and actual expert GPU kernels identified
through unique launch correlations. Symbols mark event starts, not minimum
durations. Bottom: whole-request union intersections, not estimates from the
excerpts. Blocking has zero observed overlap. Lookahead has 174.409210 ms of
read-envelope/kernel intersection (5.83314% of the read-envelope union), but
only 0.751507 ms of pread-wall/kernel intersection (0.0327443% of the pread
union). Full-trace acceptance is false; three wrapper warnings per arm remain,
and dropped-event count is unknown. Userspace wall intervals do not establish
active I/O or physical NAS-service overlap, and these profiled executions do
not establish ordinary-run speed confidence.*

This rendering-only revision leaves the original profiling/production sources
and measurement records unchanged; it is not a new experiment. Paper integration
must preserve these scope and denominator qualifications.
