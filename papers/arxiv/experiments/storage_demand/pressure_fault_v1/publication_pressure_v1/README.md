# Public cache-pressure evidence, v1

This is a **CPU-only publication projection** of the completed run
`../artifacts-go-v2-20261015-02`, not a new experiment or permission to run one.
It leaves the sealed raw receipts, protocol, production sources, manuscript,
and other publication bundles unchanged. No GPU query, server, HTTP request,
fault test, or profiling capture is executed by these scripts.

## Result and claim boundary

**3/3 fresh blocking lifecycles; 12/12 successful HTTP requests; 9/9 exact
 eight-ID canaries; 78/78 generated tokens (72 oracle-covered).**
Failed/censored/unissued: **0/0/0**. Each point completes Hello2 (two tokens,
length/usage smoke only), QA, arithmetic, and Chinese (eight actual worker IDs
and exact HTTP text/usage each). Every point therefore generates 26 tokens.
There are no extra requests, warmups, retries, replacements, or cache flushes.

| Point | Requested entries / cache MiB | Ending entries | Ending charge MiB | Owner logical peak MiB | Sampled GPU max MiB | Sampled tree PSS max MiB |
|---|---:|---:|---:|---:|---:|---:|
| small | 8 / 64 | 8 | 32.003 | 41.554 | 3,997 | 6,417.381 |
| middle | 64 / 320 | 64 | 200.023 | 209.574 | 4,125 | 6,417.131 |
| big | 128 / 640 | 128 | 392.047 | 401.598 | 4,317 | 6,419.145 |

Cache charge includes the fixed 8 MiB scratch reservation. Ending charge,
owner peak, requested ceiling, and sampled GPU memory are **parallel quantities,
not additive partitions**. Quarantine is a subset of charged bytes, not an
additional partition. A byte ceiling is neither observed occupancy nor a
physically admitted allocation.

At **every** point, the 30 decoder invocation deltas reproduce:

| Quantity | Per-point observation |
|---|---:|
| Logical checkpoint reader bytes | 36,240,064,128 |
| Checkpoint read calls | 69,114 |
| Checkpoint source checks | 46,076 |
| Verified expert reads / expert uploads | 11,519 / 11,519 |
| Expert hits | 0 |
| Artifact uploads / bytes | 34,557 / 36,240,064,128 |
| H2D copies / bytes | 88,763 / 36,240,248,688 |
| Host read logical peak reservation | 16,777,984 bytes |
| Host read final logical charge | 0 bytes |

Evictions are **11,511 / 11,455 / 11,391**. Installs minus evictions exactly
explains the **8 / 64 / 128** ending resident entries. Fewer evictions here do
**not** demonstrate cache reuse or fewer reads: reader traffic is flat and
observed expert hits are zero. Miss counts, hit rates, unique selected expert
IDs, physical source-extent counts, and physical NAS backend bytes remain
unreported/null; they are not inferred from other counters.

This is **one fixed-order coupled entry/byte-budget sweep**, not isolated LRU
sensitivity, independent byte-budget sensitivity, a working-set-fit result,
a physical capacity frontier, a statistical replication study, or causal
speedup evidence. There is no 256-entry measurement in this study.

### Memory and admission

- GPU/RSS/PSS values are maxima of **36 retained samples per point**, not exact
  lifetime peaks or proven upper bounds. All host samples were complete.
  GPU attribution is still the exclusive-operator allocation assumption;
  host/local NVML PID translation and absence of foreign interference are not
  independently proven.
- The host 64 MiB setting is a **logical read reservation** including metadata
  8 MiB and requested worker stack 2 MiB. Its measured logical high-water mark
  is below this setting; it is not a 64 MiB physical process-memory limit.
  OS/TLS/allocator overhead is excluded. Final logical zero does not establish
  physical owner release.
- Actual full-40-layer startup reports contain the requested 8/64/128 entries
  and 64/320/640 MiB limits. Ready and all four HTTP requests complete at every
  point. Physical admitted entry/byte quantities remain **null**. Neither
  successful startup nor ending occupancy fills those fields.
- The prespecified existing CPU header receipt gives a maximum whole-expert
  payload of **3,146,112 bytes** (three FP8 matrices plus BF16 scales).
  `floor((cache_bytes - 8 MiB) / payload)` is **18 / 103 / 210**; applying the
  entry cap yields logical positive-payload bounds **8 / 64 / 128**. These are
  arithmetic bounds, not occupancy, admission measurements, or a newly run
  all-cap CLI lowering test. The existing header preflight was for the middle
  setting. `summary.json` retains the prespecified geometry and its scope.
- Fixed settings include device 7 GiB, workspace 2 GiB, margin 512 MiB, KV
  16 MiB, context 128, two sequences, batch/prefill 8, cohort 1/0, TP1/PP1/EP1,
  lazy experts, resident KV, NVTX off, and blocking reads. Host retention zero
  does not establish an empty OS page cache.

### Storage scope

The shared NFS client's request-phase `normalreadbytes` increases by
**36,491,264,808 per point**. `serverreadbytes` and read-RPC counters have zero
delta. The public record retains timestamped before/after mount-counter
projections for all request and lifecycle phases so these deltas can be
recomputed. These are shared **client** counters, not exclusive process
attribution or physical NAS backend disk traffic. Cold-cache state and actual
backend bytes remain unknown. The application reader's logical bytes and the
NFS client's normal-read bytes are distinct observations.

### HTTP timing and helper cost

| Point | Hello2 E2E s | QA E2E s | Arithmetic E2E s | Chinese E2E s | Point elapsed s | Request phase s |
|---|---:|---:|---:|---:|---:|---:|
| small | 2.091585 | 7.651030 | 7.432687 | 7.080052 | 107.126 | 64.443 |
| middle | 1.904555 | 7.460827 | 7.211351 | 6.817563 | 106.004 | 63.709 |
| big | 2.137180 | 7.643562 | 7.515185 | 6.992821 | 107.054 | 64.409 |

These are **individual** observations, n=1 per prompt/point. Four different
prompts are not four statistical replicates. No pooled latency, confidence
interval, or causal speedup is reported. E2E is client send through full
response EOF, not a helper span; TTFT is the first nonempty SSE content event,
not an exact token boundary.

The 1,074 timing start/end records retain 537 completed named spans. Each
request has three candidate-pin rehash checks, approximately three seconds
each, **outside HTTP E2E**. The roughly 9–10 seconds of validation/observation
per request are retained in the request phase, not deleted to improve latency.
`summary.json` preserves measured inclusive wall and thread-CPU totals and
per-request helper windows. Nested spans and concurrent observer spans must
**not** be added across names/threads. Historical helper timings stay null.

Study elapsed through the final result is **362.815100 s**, within 900 s;
each point is within its 240 s cap. This is not elapsed through the later final
idle handoff. The separate final handoff records no signals, extra server
starts, requests, or profiling captures.

## Quality, provenance, and lifecycle evidence

- `records.jsonl.gz` contains actual HTTP completion IDs, all raw SSE events,
  HTTP completion/usage, worker admissions/terminal gates, and all 78 actual
  token-enqueue IDs. The checker derives the `cmpl-ferrule-N` to worker-ID join
  and then checks canonical eight-ID/text references. It does not retokenize
  output or infer IDs from SSE event counts. The arithmetic continuation is
  the recorded oracle, not a substituted human answer.
- The executed CLI86 candidate is commit
  `86c58a017d7c4b53490f536a3b1a133f2f7848cc`, binary SHA256
  `7cc8e0ff6e84f27090494b8e9e27dc3a388d1d2f3fc34ae39dae416e1e607a01`.
  The source digest and 1,484-member archive identity remain in the bundle.
  The pressure source-role binding commit is
  `f3472c10656d255f783c1d8035fd7167bb439e4b`, **not** a rebuilt CLI identity.
  Both live/frozen Rust discrepancies are retained as provenance; neither
  live file was consumed by the compiled candidate. Host helpers stayed
  live-pinned; the full rehash checks were not bypassed.
- Prior nine-gate full numerical qualification is **middle-only historical
  evidence**. The new pressure runs establish only the prespecified visible
  HTTP/ID scope. Full-logit/route/state equivalence is unverified at every new
  pressure point; no full numerical PASS is transferred to small or big.
- Ready and after-request library checks each cover 12 pinned libraries, with
  one ignored deleted non-library mapping. Pathname SHA256/inode evidence is
  not a hash of live mapped memory.
- The retained records support authenticated subreaper/ECHILD drain, absent
  recorded owners, three consecutive all-eight idle samples at the checked
  boundaries, local lock release/reacquisition, and the final idle handoff.
  The owner closure still reports resident cache and **unobserved physical
  allocator/release state**. ECHILD, idle, and logical zero do not prove
  universal allocator release. This CPU publication pass did not perform
  fresh idle checks or take GPU locks.

## Separate, sticky history

`history.json` keeps predecessor outcomes separate; it never contributes
requests, tokens, timing, memory, or counters to the three-point figure.

1. Dialogue-only/pre-execution zero-run states are not executed experiments.
   No receipt or zero-cost runtime measurement is manufactured for them.
2. `artifacts-go-20261015-01` remains a permanent STOP: **3 successful,
   1 censored, 8 unissued** of 12 scheduled requests; zero qualifying points.
   All 26 worker IDs were enqueued and three raw eight-ID sequences matched,
   but the Chinese request lacked complete HTTP terminal/usage evidence.
   Only **two HTTP exact canaries** passed. It had **6.873322 s** remaining
   under the old aggregate deadline. Raw token coverage does not upgrade the
   censored outcome. The original receipt and STOP marker are not rewritten.
   Observed historical gaps are not retrospectively assigned helper costs.
3. `artifacts-go-v2-20261015-01` remains the bound-source admission rejection:
   **0 server starts, 0 issued, 12 unissued**; no registration or study clock.
   Actual caps, latencies, counters and memory are unobserved, not zero.
4. Only `artifacts-go-v2-20261015-02` supplies this complete comparison. It is
   a distinct role-bound study, not a resumed or pooled successful replacement.

## Files and reproducibility

| File | Role |
|---|---|
| `curate.py` | Offline allowlisted extraction; rehashes all 244 original files before writing only this public bundle |
| `verify.py` | Standard-library checker; derives counters, gauges, samples, actual HTTP/token joins, timing and custody checks |
| `test_verify.py` | Positive and adversarial CPU tests for incomplete/tampered evidence |
| `records.jsonl.gz` | Deterministic gzip, typed JSONL with source receipt/line bindings; no repeated nested lifecycle reports |
| `summary.json` | Human-readable verified scalars and per-request helper totals; figure input is independently rederived |
| `history.json` | Separate stopped-study outcomes and historical limitations |
| `source-manifest.json` | Byte-for-byte original 244-file SHA256 inventory, not a fabricated replacement seal |
| `manifest.json` | Compressed/uncompressed public-data hashes, sizes and supplemental plan/history hashes |
| `SCHEMA.md` | Record types, transformations and aggregation rules |
| `requirements-plot.txt` | Plotting dependency only; the evidence checker needs no third-party package |

From the repository root:

```sh
python3 -B papers/arxiv/experiments/storage_demand/pressure_fault_v1/publication_pressure_v1/verify.py
python3 -B -m unittest discover -s papers/arxiv/experiments/storage_demand/pressure_fault_v1/publication_pressure_v1 -p 'test_*.py' -v
python3 -B papers/arxiv/figures/cache-pressure-budget.py
```

The last command needs Matplotlib; it writes only the new
`papers/arxiv/figures/cache-pressure-budget.pdf` and `.png`. It uses the Agg
backend and never executes model/GPU code. The source is
`papers/arxiv/figures/cache-pressure-budget.py`; the full caption is
`papers/arxiv/figures/cache-pressure-caption.md`. No manuscript integration is
performed here.

For a custodian with the original private evidence, also run:

```sh
python3 -B papers/arxiv/experiments/storage_demand/pressure_fault_v1/publication_pressure_v1/verify.py --check-source papers/arxiv/experiments/storage_demand/pressure_fault_v1/artifacts-go-v2-20261015-02
python3 -B papers/arxiv/experiments/storage_demand/pressure_fault_v1/publication_pressure_v1/curate.py
```

The source manifest's SHA256 is
`c570b05aa5c6ffc63c588c7e78441728cf1c1db9119e83a1a9a3c813efb7a283`.
Public projection hashes are **not** hashes of the original receipt bytes.
The inventory provides content-addressed provenance and the local rehash
check; it is not an independent external attestation, an archive of all 244
private files, or proof of physical resource attribution.

Validation was performed with Python 3.10.9 for extraction/checks/tests and
Python 3.12.4 / Matplotlib 3.8.4 for rendering. See `VALIDATION.md` for the
final handoff. `/papers/` is ignored by the repository; nothing is staged or
committed by this work.
