# Ferrule artifact guide

This is the reproducibility entry point for **Ferrule: Bounded Storage-Backed
Weight Residency**. Scientific labels identify results; directory names and
source revisions below identify their engineering provenance. The manuscript
need not reproduce those paths, hashes, selectors, or collection histories.
Read the [claim–evidence ledger](EVIDENCE_LEDGER.md) for scientific boundaries and
the [completion checklist](PAPER_COMPLETION_CHECKLIST.md) for remaining work.

## Availability and trust boundary

The [project repository](https://github.com/devillove084/Ferrule) contains the
curated inputs linked below. This is not a DOI-backed deposit or a promise of
permanent availability of external data. Use three distinct reproduction levels:

| Level | Required inputs / environment | What it establishes |
|---|---|---|
| **Public receipt/data verification** | Public bundles and Python standard library; Python 3.10+ for archive/platform tests, 3.11+ for the HTTP import chain | Integrity, internal relationships and reproducible arithmetic; not replay of private authentication |
| **Public figure regeneration** | Verified public data plus the pinned Python/Matplotlib/dependency/font environment | Recomputed measurements and deterministic figure bytes in that environment; no GPU or private raw tree |
| **Native rebuild and GPU serving/re-execution** | External pinned source/build/native artifacts, model/oracle payloads, runtime and fresh owner authorization | A separately qualified execution; neither receipt checks nor plotting supplies these inputs |

Passing a public checker does not remeasure a GPU result or authorize another
experiment. External-input requirements are detailed below.

Public bundles use selected fields and content hashes, not raw-directory dumps.
Do not publish complete environment maps, credentials, launch authorization,
private process maps, model weights/oracle tensors, or large profiler databases
and reports. Retain such originals privately; publish only reviewed projections.
Some historical plans retain local paths and device identifiers: these are
provenance, not portable configuration. Original and projected file hashes are
different identities. SHA-256 binds bytes, not signatures, availability of omitted
payloads, or live mapped-memory equivalence. Missing measurements remain null.

## Result → input → check → figure

Paths in links are relative to this guide. Commands below run from the repository
root. A public receipt is not automatically a self-contained re-execution kit.

| Scientific label / actual result | Public input | Offline check | Figure / scope boundary |
|---|---|---|---|
| **persistent-worker**: nine completed GPU gates | [Completed stages](experiments/storage_demand/paired_read_v1/engineering-20261016-v1/completed-stages.json), [source pins](experiments/storage_demand/paired_read_v1/engineering-20261016-v1/source-pins.json) | [Bounded-summary checks](#bounded-worker-and-transfer-summary-checks) | One worker creation, 159/2,559 reuses at selected live endpoints; 10 MiB idle charge, zero live joins; final-owner join unobserved |
| **grouped-quality**: 14 completed GPU gates (backend 1, model 5, composition 8) | [Quality summary](experiments/storage_demand/paired_read_v1/engineering-20261016-v1/grouped-quality-results.json) | [Bounded-summary checks](#bounded-worker-and-transfer-summary-checks) | Actual numerical/path tests, not 14 serving lifecycles or a new performance population |
| **grouped-whole-test**: eight completed unprofiled pairs, four per read schedule | [Timing summary](experiments/storage_demand/paired_read_v1/engineering-20261016-v1/grouped-performance-results.json) | [Bounded-summary checks](#bounded-worker-and-transfer-summary-checks) | Approximately 3.9% lower paired whole-test time; the 24-job stage **failed** at profiling, not 24/24; not HTTP latency |
| **grouped-api-observations**: qualified selected-process pairs for both schedules | [Profile summary and qualified pair index](experiments/storage_demand/paired_read_v1/engineering-20261016-v1/grouped-profile-results.json) | [Bounded-summary checks](#bounded-worker-and-transfer-summary-checks) | 382,743 → 257,163 native stream-sync rows, −5 per logical upload; partial observations, full trace false, drops unknown |
| **http-transfer-acceptance**: four fresh servers, 48/48 checked requests | [Acceptance summary](experiments/storage_demand/paired_read_v1/http-transfer-v1/acceptance-results.json) | [Bounded-summary checks](#bounded-worker-and-transfer-summary-checks) | Quality-only acceptance; all timing aggregates null, not pilot repetitions |
| **http-transfer-pilot**: eight fresh servers, 96/96 requests; 40 measured QA8, 320 measured tokens | [Pilot summary](experiments/storage_demand/paired_read_v1/http-transfer-v1/pilot-attempt-03.json), [all 96 request records](experiments/storage_demand/paired_read_v1/http-transfer-v1/pilot-attempt-03-requests.json) | [Bounded-summary checks](#bounded-worker-and-transfer-summary-checks) | Two pairs per read schedule, grouped/synchronous; lookahead benefit and blocking TTFT regression; phase goodput includes gaps |
| **retained-upload negative result**: eight completed unprofiled pairs | [Timing summary](experiments/storage_demand/paired_read_v1/engineering-20261016-v1/performance-results.json), [partial profiles](experiments/storage_demand/paired_read_v1/engineering-20261016-v1/profile-results.json) | [Bounded-summary checks](#bounded-worker-and-transfer-summary-checks) | +54.51%/+56.49% whole-test time, blocking/lookahead; rejected historical alternative, not a supported current production policy |
| **factory-qualification (earlier serving cut)**: nine gates | [Build and quality projection](experiments/storage_demand/paired_read_v1/publication_results_v1/design/source-and-build.json) | [Paired checker](#public-cpu-checks), projection only; [external originals](#external-qualification-for-the-earlier-serving-cut) | Qualification table, not a performance figure; complete raw qualification for that cut is not a portable public archive |
| **earlier-factory-qualification**: nine gates on a different source | [Qualification archive](experiments/storage_demand/qualification_v1/README.md) | [Qualification checker](#public-cpu-checks) | Historical qualification only; not another nine serving-source repetitions |
| **paired-study (earlier read scheduling)**: five completed pairs, ten lifecycles, 120 completed requests; 50 measured QA | [Paired receipts and results](experiments/storage_demand/paired_read_v1/publication_results_v1/README.md) | [Paired checker](#public-cpu-checks) | [Latency](figures/paired-read-latency.pdf), [read wait](figures/paired-read-read-wait.pdf), [accounting](figures/paired-read-accounting.pdf); [public figure regeneration](#paired-public-figures) |
| **cache-pressure**: three completed blocking lifecycles; 12 completed requests | [Pressure projection](experiments/storage_demand/pressure_fault_v1/publication_pressure_v1/README.md) | [Pressure verifier](#public-cpu-checks) | [Cache budgets](figures/cache-pressure-budget.pdf); one fixed-order coupled entry/byte sweep, not a capacity frontier or repeated timing study |
| **scoped-overlap**: two accepted profiled QA arms | [Interval archive](experiments/storage_demand/profile_overlap_v1/publication_v1/README.md) | [Overlap checker](#public-cpu-checks) | [Read/compute overlap](figures/read-compute-overlap.pdf); scoped acceptance true, full-trace acceptance false, dropped-event count null |
| **mixed-fault**: two selectors / four actual component cells pass | [Portable actual-fault archive](experiments/storage_demand/pressure_fault_v1/publication_fault_v1/README.md), [typed evidence](experiments/storage_demand/pressure_fault_v1/publication_fault_v1/SCHEMA.md) | [Fault checker and 27 tests](#public-cpu-checks), no private inputs | [Fault custody](figures/fault-custody-contract.pdf); tiny Mixed CUDA fixture, not the serving executable, a 35B fault experiment or an integrated broker ledger |
| **http-cancel**: one lifecycle, two requests; one controlled censor and one exact recovery pass | [Public receipt](experiments/storage_demand/http_cancel_v1/receipts/result.json), [results](experiments/storage_demand/http_cancel_v1/RESULTS.md) | [HTTP receipt tests](#http-receipt-check) in a repository checkout | Cancellation/recovery table, no performance figure; cancellation is not an eight-token QA success or proof that a synchronous read was interrupted |
| **platform-inventory (post hoc)**: CPU/kernel-visible memory recorded later on the paired-study boot | [Inventory receipt and scope](experiments/platform_inventory_v1/README.md) | [Offline inventory tests](#platform-inventory-offline-check) | Methods metadata, not a new performance or capacity observation; [same-boot caveats](#post-hoc-platform-inventory) apply only to paired-study |

Detailed scalar tables live in those result documents, not in parallel copies
here. Requests, paired blocks, component cells, and distinct source cuts must not
be pooled into a larger sample size.

## Worker and transfer evidence: separate comparisons

The newer [worker/upload evidence](experiments/storage_demand/paired_read_v1/engineering-20261016-v1/README.md)
and [HTTP evidence](experiments/storage_demand/paired_read_v1/http-transfer-v1/README.md)
are **curated derived summaries**, not raw public archives. Their manifests bind
the curated files; embedded hashes identify ignored local reports, traces,
SQLite, executables, source archives and custody receipts. Those originals,
model/oracle payloads and build dependencies are not supplied by the manifests.
The HTTP README's historical “pending main review” wording describes curation
handoff, not an additional scientific outcome or a public replay guarantee.

- **Persistent worker:** nine actual gates on their own source cut. Full-forward
  endpoints show one creation and 159 reuses; fresh/no-observer endpoints show
  one creation and 2,559 reuses. The live idle worker retains 10 MiB (8 MiB
  metadata + 2 MiB requested stack), with zero joins while alive. Final-owner
  join and universal physical release remain unobserved. Do not infer either
  from ECHILD or substitute later worker counters for the earlier five-pair data.
  The same completed-stages file separately records one backend ticket-API gate
  and 11 retained-upload integration gates on older cuts; these are not another
  persistent-worker denominator or current grouped-CLI qualification.
- **Grouped quality and whole-test timing:** 14 gates pass, comprising backend 1,
  model 5 and composition 8. The separate fixed 24-job timing stage passes four
  primaries and 16 unprofiled trials, then fails at its first profile; three jobs
  remain unissued. The eight completed pairs have median grouped/synchronous
  whole-test ratios 0.961331 blocking and 0.961096 lookahead. Whole-test time
  includes construction, reads, checks and teardown; it is not model-only,
  per-request, token or HTTP latency. The stage is never relabeled 24/24.
- **Grouped API observations:** use `qualified_schedule_pair_index` in the
  profile JSON, not just its original root status. The original four-profile
  stage remains failed; its qualified blocking pair uses control `d48cad78`.
  A separate two-profile lookahead stage passes under control `99c6b92`, using
  the same frozen application. Each within-schedule pair has matched controls;
  no arm is spliced across stages. Both observe native `cuStreamSynchronize`
  rows 382,743 → 257,163, a reduction of five per 25,116 logical uploads.
  H2D counts/bytes and observed kernel counts match. These are partial selected-
  process counts, not expert-range attribution, exhaustive API counts, a full
  trace, or a causal elapsed-time saving; dropped-event counts remain null.
- **Retained-upload alternative:** the earlier same-cut experiment records
  blocking/lookahead median whole-test regressions of 54.51%/56.49%. Separate
  partial profiles observe about 29.85/29.47 s of retained pin-plus-free host
  API sums. These sums may overlap other work and do not exactly explain the
  elapsed regression. The rejected production path is not retained as a
  supported mode of the cleaned engine. Preserve its experimental evidence
  and Git history, not a copied or vendored dead implementation. Historical
  reproduction requires its pinned Git source and external build inputs;
  historical flags/selectors are not promised to run on current HEAD.

### Current production selection versus measured policies

After production cleanup (`080a903d96151d02c54b28727ff3f8ffa571ea25`), ordinary
serving defaults are blocking reads with synchronous transfer. When transfer
policy is omitted (`None`), an explicitly bounded lookahead pipeline resolves
to grouped synchronous only on the supported local resident Mixed BF16/F32 CUDA
numeric capability; otherwise resolution remains synchronous. Explicit
synchronous selection remains the ablation baseline. Public serving rejects
explicit grouped transfer with blocking or absent bounded reads; grouped/blocking
is a private experimental cell, not a public serving mode. The regressed retained
upload implementation and production selector are removed. Current selector
validation is CPU-only, not post-cleanup GPU, HTTP, numerical-quality or
performance requalification. The frozen pilot did not benchmark the post-cleanup
resolver. The frozen application and control identities below remain the
provenance of all reported runtime results.

### HTTP comparison and metric boundaries

Acceptance is four actual fresh servers and 48 correct requests, including
20 measured QA8 requests / 160 measured tokens. It is quality evidence only.
The later completed pilot is a separate eight-server, four-pair comparison:
96 requests, 40 measured QA8 requests and 320 measured tokens. Every scheduled
request completes correctly; failed/censored/unissued counts are zero. Normal
EOS stopping, payloads, seed and references are bound in the summaries.

The pilot compares **grouped-synchronous / synchronous transfer within each
read schedule**. Two paired blocks per schedule give descriptive median
ratios; cross-block independence is not established under shared cache state:
blocking client TTFT **+4.96%**, E2E approximately flat (−0.43%), phase goodput
+3.74%; lookahead TTFT −8.47%, E2E −7.34%, phase goodput +6.52%. These frozen
explicit-policy comparisons did not benchmark the post-cleanup resolver and
do not establish significance or a universal improvement. The current selection
rules above have CPU-only validation, not new GPU qualification. Do not pool
requests as independent replicates, combine read schedules, or reuse the older
five-pair read-scheduling estimates as these transfer-policy effects.

Client TTFT is first nonempty SSE content minus client send time, not first
GPU-token latency. E2E is client terminal minus send time. Phase width is the
last measured client terminal minus the first measured client dispatch,
**including inter-request gaps**; phase-qualified goodput is qualified measured
output tokens divided by that width. The separately retained request-only
serialized-service rate excludes those gaps and is not the same metric. The
older paired study's serial-window goodput also has its own definition.

Each pilot arm records 40,943 uploads and 128,811,927,312 logical H2D bytes.
Lookahead reports 36,863 speculative/taken reads; blocking reports zero. Final
live-reader charge remains 10 MiB, with observed logical host peak 16,777,984 B
below 64 MiB. Requested/effective policy getters, positive uploads and pinned
source dispatch support path evidence, not a direct grouped-backend-call
counter, route/logit measurement, physical NAS-byte measurement, or full
resource-contract proof. Planned model returns, ACK ordering, authenticated
ECHILD, strict idle and lock reprobes are not CUDA-release certificates.

Pilot attempts 01 and 02 remain permanent **zero-server/zero-request** setup
failures, with 96 requests and 40 measured QA slots each unissued. Attempt 01
fails absolute reserve admission; attempt 02 reaches the active setup deadline.
One launch journal is not a server start. Later recovery/cleanup does not heal
those outcomes. The third pilot uses separately reviewed role-partitioned
controls: full build/input identity is proved once, while 145 live-role inputs
are freshly checked at entry (18,377 build-only and 589 archived-quality roles
are separately bound). Its static preparation is about 1.84–2.16 s, with no
larger timers. This is admission/control engineering, not a model-speed result.
The exact interrupted operation in the earlier attempts remains unknown.

### Bounded worker and transfer summary checks

This standard-library-only check verifies the two curated manifests without
opening local raw evidence, querying a GPU, rebuilding, or running a server.
It does **not** independently rederive the local raw measurements or authorize
historical commands. Run from the repository root:

```sh
python3 -S -B - <<'PY_CHECK'
import hashlib
import json
from pathlib import Path
base = Path("papers/arxiv/experiments/storage_demand/paired_read_v1")
for name in ("engineering-20261016-v1", "http-transfer-v1"):
    root = base / name
    manifest = json.loads((root / "manifest.json").read_text())
    files = manifest["files"]
    rows = files.items() if isinstance(files, dict) else ((r["path"], r) for r in files)
    for path, record in rows:
        data = (root / path).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["sha256"], path
        assert len(data) == record.get("bytes", record.get("size_bytes")), path
    print(name, "curated manifest PASS")
PY_CHECK
```

The exact historical build/test/public CLI argv are data in the source-pin and
pilot records, not commands to replay a consumed stage. Sealed/fake/no-op
preflight plans or plans pending main GO do not establish actual GPU execution
and are not public re-execution instructions. New execution needs new output
roots, required private/external inputs, independent review and fresh owner GO.
Do not rerun stopped generations or the failed profile stages.

## Public CPU checks

Use **Python 3.10+**, standard library only, for the five archive checks and
platform-inventory tests below.
The inspected environment resolves `python3` to `/opt/conda310/bin/python3`
(version 3.10.9); check `python3 --version` on another host. Do not use an
unqualified `python`, which can resolve to Python 2. The overlap checker alone
also documents Python 3.9 support. These commands neither query a GPU nor launch
a server, access model weights, download data, or overwrite sealed inputs.

```sh
python3 --version
python3 -S -B papers/arxiv/experiments/storage_demand/qualification_v1/offline_check.py
python3 -S -B papers/arxiv/experiments/storage_demand/qualification_v1/test_offline_check.py

python3 -S -B papers/arxiv/experiments/storage_demand/paired_read_v1/publication_results_v1/check.py
python3 -S -B papers/arxiv/experiments/storage_demand/paired_read_v1/publication_results_v1/test_publication.py

python3 -S -B papers/arxiv/experiments/storage_demand/pressure_fault_v1/publication_pressure_v1/verify.py
python3 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/pressure_fault_v1/publication_pressure_v1 -p test_verify.py -v

python3 -S -B papers/arxiv/experiments/storage_demand/profile_overlap_v1/publication_v1/check.py
python3 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/profile_overlap_v1/publication_v1 -p test_publication.py -v

python3 -S -B papers/arxiv/experiments/storage_demand/pressure_fault_v1/publication_fault_v1/check.py
python3 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/pressure_fault_v1/publication_fault_v1 -p "test_*.py" -v
```

The complete qualification, paired-results, pressure-publication, scoped-overlap
and mixed-fault publication directories provide their own checking inputs.
Their checkers verify file membership/hashes and evidence-specific relationships;
they do not establish the authenticity of withheld originals. A detached copy of
the fault archive passes its checker and all 27 tests without a repository,
source tar, native executable, model, `target/` tree or private evidence. Its
optional companion-file check also needs only public files:

```sh
python3 -S -B papers/arxiv/experiments/storage_demand/pressure_fault_v1/publication_fault_v1/check.py --figures papers/arxiv/figures
```

### Platform inventory offline check

The four-file [platform supplement](experiments/platform_inventory_v1/README.md)
is self-contained for 29 parser/binding and receipt-consistency tests. These
checks do not read live hardware or require private originals:

```sh
python3 -S -B papers/arxiv/experiments/platform_inventory_v1/test_inventory.py
```

Do not use `inventory.py --output` for ordinary verification: that is a new
post-hoc collection requiring authorized local proc/sysfs/cgroup access and
preserved original study evidence, not a reproduction of the old measurement.

### HTTP receipt check

This check needs a repository checkout, including the shared experiment helpers
and `scripts/`, rather than just the `http_cancel_v1` directory. It imports
`tomllib`: **Python 3.11+ is required**, and Python 3.12.4 was tested. The literal
interpreter below is available on the inspected host; elsewhere use the path to
a Python 3.12 installation, changing only the interpreter prefix.

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/http_cancel_v1 -p test_receipt.py -v
```

These five tests check the public projection and rejection of a synthetic private
credential. They pass from a temporary export of only tracked experiment/helper
sources and scripts, without the private case. They do not rederive omitted raw
HTTP logs or replay launch authentication. `curate.py check` is instead a
**custodian-only** comparison requiring the preserved private evidence.
`test_http_cancel.py` additionally exercises localhost HTTP client behavior;
it is not necessary to inspect the published receipt.

## Figure reproduction

Figure generation is CPU-only but needs Matplotlib; do not pass `-S` when using
installed plotting packages. The recorded renderer is Python 3.12.4,
Matplotlib 3.8.4, NumPy 1.26.4, Pillow 10.3.0 and FreeType 2.12.1, with hashed
DejaVu Sans fonts. Exact records, source hashes, and output hashes are in the
[paired manifest](experiments/storage_demand/paired_read_v1/analysis_v1/plot-manifest.json),
[overlap figure record](experiments/storage_demand/profile_overlap_v1/publication_v1/figure.json),
and [fault figure record](experiments/storage_demand/pressure_fault_v1/publication_fault_v1/figure.json).
Byte equality is not promised across arbitrary renderer installations.

- **Scoped overlap:** all timing/attribution inputs needed to render are public.
  The following read-only check recomputes interval unions, renders in memory,
  and compares PDF/PNG/metadata bytes. See its [guide](experiments/storage_demand/profile_overlap_v1/publication_v1/README.md)
  for separate-output rendering and layout tests.

  ```sh
  /home/admin/wxw403883/anaconda3/bin/python3.12 -B papers/arxiv/figures/read-compute-overlap.py --check
  ```

- **Cache pressure:** the generator rederives its values from the public records.
  Render to a new output directory rather than overwrite tracked assets:

  ```sh
  mkdir -p target/reproduction/cache-pressure
  /home/admin/wxw403883/anaconda3/bin/python3.12 -B papers/arxiv/figures/cache-pressure-budget.py --output target/reproduction/cache-pressure
  ```

  The [dependency pin](experiments/storage_demand/pressure_fault_v1/publication_pressure_v1/requirements-plot.txt)
  and [caption](figures/cache-pressure-caption.md) document rendering and scope.
  This generator does not provide a `--check` mode.

- **Mixed fault:** all producer records, typed projections, source assertions and
  rendering inputs are public. The [renderer](figures/fault-custody-contract.py)
  does not need the original source archive or validation directory. This
  read-only byte check needs the recorded plotting environment:

  ```sh
  /home/admin/wxw403883/anaconda3/bin/python3.12 -B papers/arxiv/figures/fault-custody-contract.py --check
  ```

  The [caption](figures/fault-custody-caption.md) separates serialized observations
  from source-and-PASS-backed API/fence assertions. Figure-data/layout and image
  hash verification are already covered by the standard-library checker above;
  those checks do not invoke the renderer.

### Current worker/transfer figures

The two new figures use the separate
[`render_current_evidence.py`](figures/render_current_evidence.py), not the older
paired renderer or its frozen manifest:
[`current-http-outcomes.pdf`](figures/current-http-outcomes.pdf) and
[`current-transfer-mechanism.pdf`](figures/current-transfer-mechanism.pdf).
Their [current evidence manifest](figures/current-evidence-manifest.json) binds
the generator, pinned renderer/fonts, input hashes, all four PDF/PNG outputs,
and experiment-specific denominators. The check rederives the HTTP pair ratios,
checks 96 preserved raw request receipts and four full acceptance stats
snapshots, and validates figure text bounds, non-overlap and minimum eight-point
fonts. It is a custodian check requiring those hash-bound local originals;
**the curated public summaries alone are not a portable regeneration kit for
these figures**. It neither invokes the older renderer nor verifies an entire
build-input/raw-artifact tree.

Run this explicit, CPU-only, read-only verification from the repository root in
the pinned environment:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -B papers/arxiv/figures/render_current_evidence.py --check
```

To intentionally regenerate only these two PDF/PNG pairs and their manifest,
run the same command without `--check`. No GPU experiment is run. The figure
Makefile builds the mechanism diagrams; it does not regenerate these new plots
or replace their explicit check. The older public paired-figure route below
continues to use its unchanged renderer and manifest.

For label-only maintenance without reopening raw receipts, both generation and
`--check` also accept `--verified-data-manifest PATH` together with
`--verified-data-sha256 SHA256`. Use a preserved copy of a previously verified
current-evidence manifest and its exact SHA-256. This mode verifies that snapshot,
its internal evidence identity, input pins and renderer/font environment, then
reuses its plot data and prior validation results. It does **not** newly verify
the raw HTTP receipts or confer new runtime qualification. The new output
manifest binds the changed renderer; input and measurement identities stay fixed.

### Paired public figures

[`analysis_v1/render_public.py`](experiments/storage_demand/paired_read_v1/analysis_v1/render_public.py)
is the supported public reproduction entry point. It first verifies the complete,
unmodified **623-file** `publication_results_v1` archive, then reconstructs
latency, read-wait and accounting figures from public measurements. The archive
manifest remains SHA-256
`14890f2d90461c0b228a804ae8c77d3ae93cdda6417bdcd1880292839fddccd0`.
No private 840-input tree, saved plot-data file, model, source tar or Git checkout
is required. The public path does not import the private derivation modules.

For portable commands, `python3` must select the rendering environment, not the
standard-library-only checker interpreter. Use a Python 3.11+ installation with
the recorded packages; **the current exact pins require Python 3.12.4**, not an
arbitrary 3.11+ interpreter. The
[package record](experiments/storage_demand/paired_read_v1/analysis_v1/requirements-render.txt)
and [renderer pins](experiments/storage_demand/paired_read_v1/analysis_v1/renderer-pins.json)
record Matplotlib 3.8.4, NumPy 1.26.4, Pillow 10.3.0, FreeType 2.12.1, zlib 1.2.13,
transitive dependencies and DejaVu Sans font hashes. Package versions alone do
not establish the linked FreeType/zlib builds. The renderer installs nothing
and fails before plotting when the recorded environment/font pins differ.

With the already validated interpreter on the recording host:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -B papers/arxiv/experiments/storage_demand/paired_read_v1/analysis_v1/render_public.py --output-dir /tmp/ferrule-paired-read-public
/home/admin/wxw403883/anaconda3/bin/python3.12 -B papers/arxiv/experiments/storage_demand/paired_read_v1/analysis_v1/render_public.py --output-dir /tmp/ferrule-paired-read-public --check
```

Elsewhere replace only the interpreter prefix with `python3` from the pinned
environment. The explicit output directory must be outside the frozen archive;
the first command creates it and writes three PDFs, three PNGs, a preview, plot
data, layout and a manifest. The second revalidates the archive, re-renders, and
compares actual output hashes and bytes without rewriting them. Neither command
implicitly writes the repository figures or reseals the public archive.

For a detached copy, keep the complete `publication_results_v1/` beside
`analysis_v1/` containing `render_public.py`, `render.py`, `plot_common.py`,
`renderer-pins.json` and `requirements-render.txt`. The pinned Matplotlib
installation supplies the hashed fonts. Copy `tests/test_public_render.py` as
well to run the public regression suite; see the [plot guide](experiments/storage_demand/paired_read_v1/analysis_v1/PLOTS.md)
for the exact layout. No original experiment directory is needed.

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -B -m unittest discover -s papers/arxiv/experiments/storage_demand/paired_read_v1/analysis_v1/tests -p test_public_render.py -v
```

All 12 tests pass, including independently reconstructed arm/request values and
all 3,125 resamples, tamper rejection, and byte-identical local/detached outputs.
Public collector-summary maxima/counts remain sampled device-wide observations;
raw GPU sample streams are not invented or required by this renderer. Requested
host-to-device bytes remain the logical `h2d_bytes` counter, not bus traffic.

The public output manifest binds this public input identity and renderer, so its
image hashes intentionally differ from the earlier private-provenance figures.
Byte reproduction applies to the current public outputs and pinned environment,
not to arbitrary historical PDF bytes. `plot_paired_read.py` now delegates to the
public path by default; the historical private derivation is explicitly opt-in
and is not a requirement for reproducing the three current figures.

Mechanism diagrams explain the design rather than plot measurements; their
[figure guide](figures/README.md) documents sources. The four historical
`storage-demand-*.pdf` plots use the separate [historical analysis](experiments/storage_demand/analysis/README.md),
not the enabled paired study.

## Immutable identities and data pins

The **earlier** serving executable used for paired-study, cache-pressure,
scoped-overlap, and http-cancel stays bound to source
`86c58a017d7c4b53490f536a3b1a133f2f7848cc`, even when the current checkout gains
documentation, collectors, or component tests. **Current HEAD is not the measured
CLI.** The paired [build projection](experiments/storage_demand/paired_read_v1/publication_results_v1/design/source-and-build.json)
and [rebuild instructions](experiments/storage_demand/paired_read_v1/publication_results_v1/REPRODUCE.md)
contain exact build argv, runtime/native pins and source roles.

| External serving-source artifact | SHA-256 |
|---|---|
| CLI | `7cc8e0ff6e84f27090494b8e9e27dc3a388d1d2f3fc34ae39dae416e1e607a01` |
| Constructor test executable | `693fe2de0983432a12e46d4c079bef370466434ce2ac181bf4af2dffba24ef1f` |
| Composition test executable | `0a429c47c158bbe25f91730b75b47fd5cb768106f1d63d844da3a5f84fc8677f` |
| Original 1,484-member source archive | `e0d160996147820fcc64cfb29070b265b2297a6d9265a21de541ed77e8f2fa65` |

These binaries and source tar, CUDA/native libraries, model weights, tokenizer
payloads and oracle tensors are **not bundled**. Public metadata hashes do not
supply them. The original build used Rust `nightly-2026-08-01`, CUDA 12.4.99,
SM86, pinned CUTLASS/NVTX headers, and offline Cargo dependencies. Equal source
and common runtime pins do not prove mathematical equivalence of test and CLI
executables or a bitwise reproducible rebuild. Provisioning external inputs may
need network access and third-party licenses. New GPU work requires fresh
reviewed plans, output directories and owner authorization; never replay consumed
start/next commands or resume a stopped generation.

### Newer application and control identities

The HTTP acceptance and transfer pilot use a **different measured CLI** from
the earlier serving study:

| Identity role | Exact source or digest |
|---|---|
| Compiled production source | `d8c92d80dc2ec95c44ba7513198232d36f290a34` |
| Source archive/control cut | `37b2f40c2dfc02825a373e417b0d5bb428e1622e` |
| CLI SHA-256 | `c3db43b8c556a210a77f7fac29dc23c4bc9399d5e0374e6dc9fecd016625ccaa` |
| Source archive SHA-256 | `2645585f2e33dd833c2c785dab97d4f9f9090604b89353d940879593ef41d5bb` |
| Completed pilot orchestration, not a rebuilt CLI | `728df96114f06d63f40cf9a5f5f166b47c46295d` |

The [HTTP source pins](experiments/storage_demand/paired_read_v1/http-transfer-v1/source-pins.json)
retain the historical build and acceptance identity, unchanged. The
[pilot summary](experiments/storage_demand/paired_read_v1/http-transfer-v1/pilot-attempt-03.json)
separately binds the later controls, workflow, role partition, reviewed identity,
GO/layout and raw-final receipts. The declared Cargo/Rust scope is byte-equal
between the production and archive cuts; neither is the earlier serving cut.
The build was once, release/offline/locked, CUDA 12.4.99/SM86 with compiled
`cuda,nvtx-profile` and runtime NVTX disabled. The exact invocation and all quotas
are in those records. Checkpoint shard identities are stat-bound, not full
payload content-SHA256 certification; library maps do not hash live memory.

The grouped component and whole-test experiments instead use application
`ce6d4ff28b96d4057d0de6dfa858707104aad16b`, composition SHA-256
`d8c0bae4872ffbbb6828dd7f49acecb4de0f69ab32748662845ddaa6d0b30557`,
and archive SHA-256
`4e15b7301f678424878ba97c3a3df8f8533a696053d2eb58b36bb35e6895e135`.
The worker stage uses `dae07b85c647cfc418b5412078874f9e8e9bf1db`;
the rejected retained-upload experiment uses
`57e46295d3cf5725605e076eb5543f6124bd9245`. Their own source/control and executable
pins remain separate in the engineering summaries. Do not assign one global
source identity to these experiments.

**The current checkout, including production cleanup, is not either measured
CLI and is not newly compiled or GPU-qualified by this documentation update.**
To rebuild a historical experiment, first obtain its pinned Git tree in a
separate checkout and provision its recorded external inputs. Checking out a
revision does not itself supply omitted binaries, model weights, toolchains,
or authorization. Never restore rejected production code into current HEAD
merely to make an old selector available.

The following are compact integrity entry points, not a complete Git history.
Each manifest supplies member-level hashes and the original/projection distinction.

| Public input identity | Publication commit, where recorded | SHA-256 of the named manifest or receipt |
|---|---|---|
| [Worker/transfer curated manifest](experiments/storage_demand/paired_read_v1/engineering-20261016-v1/manifest.json) | Content-pinned derived summaries; raw local inputs external | `efc57f4aedc4c04e1147b446a89aa4dcfa10e5aeb58dfca08605686e398eb72e` |
| [HTTP-transfer curated manifest](experiments/storage_demand/paired_read_v1/http-transfer-v1/manifest.json) | Content-pinned derived summaries; raw local inputs external | `b67766f1313df7639bbb69c687a8641abb4e72d3f01cf9bf82114ba86b43a009` |
| [Earlier qualification manifest](experiments/storage_demand/qualification_v1/manifest.json) | `17b16fa` | `948d52e6485bea02eb0135690b3fcd9b8091d1b13e65d6b13934df01c32bbcac` |
| [Paired-results manifest](experiments/storage_demand/paired_read_v1/publication_results_v1/manifest.json) | `c624335` | `14890f2d90461c0b228a804ae8c77d3ae93cdda6417bdcd1880292839fddccd0` |
| [Public paired-plot manifest](experiments/storage_demand/paired_read_v1/analysis_v1/plot-manifest.json) | Public renderer; content-pinned | `38d5c40cc4dee0286c24e03310b7ef5d21c155c4e7cf2d88158d2e09da79a518` |
| [Scoped-overlap manifest](experiments/storage_demand/profile_overlap_v1/publication_v1/manifest.json) | `bd06735` | `3906671e772992eeff4187cfab71c3fe30eb923b924f4b1e52ac3f0be25723f4` |
| [Pressure manifest](experiments/storage_demand/pressure_fault_v1/publication_pressure_v1/manifest.json) | `9c4f30f` | `a614c503035cfcde8564ca825e571541e925c1d228c4014fe0fd9d7f605b9164` |
| [HTTP cancellation receipt](experiments/storage_demand/http_cancel_v1/receipts/result.json) | `ed3e11b` | `fcc477461283ead2f207917bfd8f05af19076172a75fda5cc3bdf0f911c14d15` |
| [Mixed-fault manifest](experiments/storage_demand/pressure_fault_v1/publication_fault_v1/manifest.json) | `6f814` | `5bd65d0706dda32185086f7c8eebf9533ab04856ff39a34d91dc974e6a367c71` |
| [Post-hoc platform receipt](experiments/platform_inventory_v1/posthoc-20261015.json) | Content-pinned supplement | `aadbd5efc254662afab7c24e318e43bc4bdc73d1de4a47b7d9c74ebccfcaa56c` |

The earlier qualification source is
`033519c3def098ce417e4b6ff0f141a9f30bddb1`, not the serving-source cut.
The paired readiness validator (`7d3bee2`) and independent entry (`ac224f7`),
overlap measurement correction (`5e4e29c`), and pressure source-role binding
(`f3472c1`) identify collection logic, not replacement CLI binaries.

## Post-hoc platform inventory

The [four-file supplement](experiments/platform_inventory_v1/README.md) contains
77,072 bytes (about 75.3 KiB), including the CPU-only collector, public receipt,
offline tests and interpretation guide. It reports
`POSTHOC_INVENTORY_SAME_BOOT`, **not experiment-time resource measurement**.

- CPU inventory records the model string **Intel Xeon Platinum 8369B** and
  **128 visible logical processors**. These are not allocated/physical core
  counts, a physical socket inventory or an attested host topology.
- `/proc/meminfo` reports **754.913 GiB kernel-visible MemTotal**:
  791,584,008 Linux kB = 810,582,024,192 bytes. This is not installed DIMM RAM,
  free memory, experiment memory allocation or a cgroup allowance.
- Resource acquisition was **2026-10-15T14:13:09.868869046Z through
  14:13:09.874909108Z**. The paired study ran **00:48:28.013551438Z through
  01:13:47.738082277Z** that day. The later inventory must not be backdated.
- All **300 original NFS phase-boundary snapshots**, 30 per arm across ten
  paired-study arms, match the inventory boot-ID hash. Those are boundary
  snapshots, not 300 additional experiments or GPU samples. The original audit
  also records 341 boot-ID occurrences including nested copies, which are not
  independent host measurements.

The receipt retains per-arm counts/time ranges, phase labels, public archive
manifest/provenance pins and digests of checked original inventories. The source
snapshot pins remain in the paired archive provenance; originals and raw boot
IDs are not disclosed. Offline tests check this recorded relationship and
collector-source identity, not independently reauthenticate withheld original
snapshots. The optional original-input audit is a distinct custodian operation.

Same-boot matching supports a platform description only under an explicit
**no relevant hotplug/reconfiguration assumption**. Installed RAM, physical
host topology, experiment-time affinity/cgroup constraints and hidden ancestor
limits remain unknown. Current collector affinity and locally unlimited cgroup
sentinels do not establish experiment allocations. This receipt binds only to
the paired study, not cache-pressure, profiling, faults or reference generation.

The large kernel-visible memory inventory does not certify available capacity
for any past process. Conversely, a **64 MiB logical staging reservation is not
a physical host-memory bound**. Neither that setting nor NFS-backed reads proves
that compact weights could not fit in host RAM, or that physical storage access
was necessary. Physical RAM insufficiency and backend storage necessity remain
unestablished.

## Configuration and observation dictionary

### Quality, schedules and budgets

The earlier full-model read-scheduling studies use Qwen3.5-35B-A3B-FP8, Mixed precision, PP1/TP1/EP1,
lazy experts and resident KV. The common middle configuration is context 128,
batch/prefill 8, two sequences, decode cohort target 1 / maximum deferrals 0;
device policy 7 GiB, workspace 2 GiB, margin 512 MiB, KV 16 MiB. The host read
reservation is 64 MiB including 8 MiB metadata and a requested 2 MiB worker
stack; one worker, queue depth one and two payload slots. The expert-cache
policy is 64 entries / 320 MiB including 8 MiB scratch. These are logical
accounting/configuration scopes, not a total physical-memory partition.
Cache-pressure couples caps at 8/64, 64/320 and 128/640 entries/MiB; all use
blocking. There is no 256-entry observation.

Both earlier paired arms enable the real read pipeline; only `--expert-read-schedule`
changes between `blocking` and `lookahead`. Ordinary timings use
`FERRULE_PROFILE_NVTX=0`; compiled NVTX support does not mean profiling was on.
The [preregistration](experiments/storage_demand/paired_read_v1/publication_results_v1/design/preregistration.json)
fixes five pairs, order B/L, B/L, L/B, L/B, B/L, 7,200 s total, 650 s/lifecycle
and a 100 s final reserve. It permits no retries, replacements, cache flushes
or adaptive extra warmups. The estimator is `exp(median(log(L/B)))` over blocks;
all 3,125 ordered paired resamples supply descriptive ranks 79/3047, not
coverage-guaranteed confidence limits. Individual requests are nested observations.

Factory numeric comparison uses `0.00025 + 0.012 * abs(reference)` and exact
IDs where observed. Each schedule retains separate common-history, free-feedback,
reset and fork denominators; ordinary routes/states remain unobserved. HTTP
quality instead joins actual worker-enqueued IDs to HTTP completions, usage and
SSE termination: text retokenization and counting SSE events are not substitutes.
The [oracles](experiments/storage_demand/paired_read_v1/publication_results_v1/design/oracles.json)
contain exact IDs and reference hashes. Two-token smoke requests are usage/length
checks, not exact-eight-token canaries.

### Resource-sampling scope

The paired resource records contain **515 samples** across ten lifecycles
(49–54 per arm); pressure records contain **108 samples**, 36 per point. These
are different from the 300 NFS boundary snapshots used for post-hoc boot binding.
The [resource observer](experiments/telemetry.py) collects and records a sample,
then waits for the configured two seconds: acquisition time is additional.
This is **not fixed two-second start-to-start polling** or continuous coverage.
Public paired rendering checks the retained collector-summary maxima/counts,
not a raw sample stream. Sampled maxima do not certify a continuous lifetime
upper bound or exact process-owned device memory.

### Event and source crosswalk

| Scientific quantity or contract | Exact implementation/projection anchor | Interpretation |
|---|---|---|
| Request admissions, tokens and terminals | `request_admitted`, `request_http_admitted`, `request_worker_token_enqueued`, `request_worker_terminal`, `request_worker_terminal_enqueue`, `request_http_terminal`; [worker](../../crates/ferrule-server/src/worker.rs), [HTTP stream](../../crates/ferrule-server/src/http.rs) | Worker enqueue is not client consumption; cancellation can leave client usage null |
| Invocation and closure accounting | `mixed_decoder_invocation`, `resource_shutdown_summary`, `demand_wait_ns`; [expert reads](../../crates/ferrule-model/src/transformer/expert_reads.rs) | Per-invocation counter deltas, owner-lifetime maxima and final gauges have distinct semantics |
| Source-qualified read and bounded lookahead | [Read plans](../../crates/ferrule-model/src/checkpoint/read_plan.rs), [source identity](../../crates/ferrule-model/src/checkpoint/source.rs), [worker](../../crates/ferrule-model/src/checkpoint/read_pipeline.rs), [CUDA owner](../../crates/ferrule-model/src/transformer/standard/cuda.rs) | Ordered plans, exact tickets and alias custody; synchronous forward, not general nonblocking inference |
| Profiled read versus compute | `read_subset`, OSRT `pread`/`pread64`, `layer.expert`, `layer.expert.upload`; [source projection](experiments/storage_demand/profile_overlap_v1/publication_v1/source.json) | Read envelope includes allocation/read, excludes source verification; unique launch/API correlation identifies actual expert kernels, not a CPU submission span |
| Logical versus storage bytes | `SourceVerified`, `proved_read_bytes`, `reader_interval_bytes`, NFS `normalreadbytes`/`serverreadbytes`; [read session](../../crates/ferrule-model/src/checkpoint/read_session.rs) | Source-verified operation bytes and shared-client deltas are not physical NAS bytes; OSRT byte fields remain null |
| Completion, credit and publication contracts | [Exact keys/leases](../../crates/ferrule-common/src/io_protocol.rs), [broker](../../crates/ferrule-runtime/src/scheduling/resources.rs), [registry](../../crates/ferrule-runtime/src/io/registry.rs), [materialization provider](../../crates/ferrule-model/src/moe/cuda_materialization.rs), [pipeline owner](../../crates/ferrule-runtime/src/parallel/pipeline/owner.rs) | Design obligations and implementation anchors, not universal registry integration or an observed five-bucket ledger |

For scoped-overlap, read-byte totals are typed requested/packed bytes backed by
the complete source-verified operation ledger. They are **not OSRT-return byte
counts**. OSRT `returnValue=0` remains an opaque encoding, not proof of a zero-byte
read, per-call success or an `ssize_t` result. Per-call `osrt_bytes` and aggregate
`osrt_read_bytes` remain null. Whole-operation `SourceVerified` evidence does not
turn the profiler schema into a per-syscall success/byte journal.

The intended credit-conservation invariant requires disjoint ownership categories
and justified transfers. The implemented broker exposes capacity, execution
reserve, in-use and high-water fields; the Mixed component fixture has **no
runtime broker**. Actual fault evidence concerns logical host and expert-cache
charges and retained ownership. Unknown/quarantined cache bytes are a subset of
charged bytes, not a fifth additive physical pool. No theorem about integrated
broker conservation is experimentally established by those observations.

## External qualification for the earlier serving cut

The nine-gate raw evidence for the earlier serving source is local/external, under
`target/validation/fixesmetadataCLIcurrent-quality-20261015-01/`.
`FINAL.json`, `qualification-plan-v2.json`, `gates/*/quality.json`, the full40
`gates/*/numeric-report/full40-bf16-profile.json` reports, and their supervisor
records are **not a complete portable archive**. The public
[source/build projection](experiments/storage_demand/paired_read_v1/publication_results_v1/design/source-and-build.json)
records all nine report/supervisor hashes, executable identities, acceptance
rows and reference pins. Its check validates that projection, not the missing
raw qualification corpus. The still-earlier public qualification archive does not fill this gap. These
receipts do not substitute for the newer worker, grouped-quality or public-CLI
acceptance stages, which have their own source cuts and denominators.

Exact selectors use the constructor name
`normal_cuda_constructor_reaches_configured_provider_for_both_model_boundaries`
and the following composition names. Expand `MODE` separately as `blocking` and
`lookahead`; the notation below is a selector inventory, **not a launch command**.

```text
model_factory::dense_bf16_tests::full40::numeric_bf16_mixed_storage_middle64_MODE_public_factory_full40_logits_routes_states_match_profile_reference
model_factory::dense_bf16_tests::full40::fresh_quality::mixed_storage_middle64_MODE_fresh_quality_retained_generation
model_factory::dense_bf16_tests::full40::fresh_quality::mixed_storage_middle64_MODE_fresh_quality_no_observer_equality
model_factory::dense_bf16_tests::full40::fresh_quality::mixed_storage_middle64_MODE_fresh_quality_cold_owner_restart
```

These are one constructor plus eight schedule-specific gates. Authenticated
ECHILD, process drain and idle evidence are not post-drop device-release proof.

## Portable mixed-fault and external build inputs

The [public archive](experiments/storage_demand/pressure_fault_v1/publication_fault_v1/README.md)
preserves two actual selector executions and all four passing cells. Its two
40-row producer JSONL reports and matching stdout/stderr are byte-identical to
pinned originals. The checker rederives `postfailure.json` in memory; it does not
accept that derived summary as another measurement. The public archive plus its
four companion figure files comprises **36 files / 1,564,361 bytes** (1.56 MB,
1.49 MiB), listed in its
[exact inventory](experiments/storage_demand/pressure_fault_v1/publication_fault_v1/STAGE_PATHS.txt).
The payload manifest separately covers 19 data/reference files / 839,845 bytes;
it does not include its own bytes, the checker or the separately pinned figure.

No original validation directory, source tar, checkout branch, CUDA installation
or private input is needed for public checks. The archive includes nine pinned
source-reading references and a complete **1,487-file hash inventory**, not all
1,487 source files or a standalone native build tree. Private custody originals
remain external; the safe custody projection is an audited conclusion, not an
independently replayable authentication handshake.

### Serialized observations versus executed assertions

The [evidence schema](experiments/storage_demand/pressure_fault_v1/publication_fault_v1/SCHEMA.md)
provides source and report-row navigation. Its distinctions are part of the
scientific claim, not merely publication formatting:

- **Serialized:** actual fault boundary, successful original compute/upload wait,
  deliberate owner-proof discard, metadata/charge/custody snapshots, transaction
  counts and host-read deltas. Snapshot sampling does not read device contents.
- **Serialized assertion outcome:** clean committed-content equality is backed
  by the pinned bitwise KV/convolution/SSM comparisons plus PASS. Raw state bits
  are not published. Post-Unknown content equality and raw contents remain null.
- **Source + actions + PASS:** two later independent compute and two upload waits,
  two typed Unknown aborts, rejected reset/admission/KV release/publish/shutdown,
  and healthy clean tx3. These are not separately serialized per-call results.
  Only execution and abort assert typed `CompletionUnknown`; the other rejected
  APIs assert errors without the stronger type claim.
- **Release:** only clean records fixture allocator-requested bytes after explicit
  owner drop and retirement trim. Unknown intentionally retains owners until
  process exit. Neither process drain nor device idle discharges that custody.

The tested component source is `79d684f5c79fc6c3961161111ae6efd1a37ea8ea`,
archived from `bd0673579286e614079ccfe5d972be19f860e9e8`; the pinned handoff records
selected-component-source equivalence. The distinct test executable has SHA-256
`e531be47a5113cc53665fdc6576cf389b56c29cb7e16f275efec52b0f79b6903`;
its 1,487-member source tar has SHA-256
`95423c59156abc391eb0a3d80c3c1b2c368bddb5366931456d0bebf1fdd18983`.
The [source manifest](experiments/storage_demand/pressure_fault_v1/publication_fault_v1/source-manifest.json)
binds those external build inputs. Neither executable nor tar is bundled, and
the serving CLI was not rebuilt or requalified by these tests.

Exact selectors, in the
[archived component source](experiments/storage_demand/pressure_fault_v1/publication_fault_v1/reference/fault_cuda.rs):

```text
transformer::forward::tests::cuda::mixed_fault::mixed_bf16_clean_fault_after_real_expert_submit_both_schedules
transformer::forward::tests::cuda::mixed_fault::mixed_bf16_unknown_after_real_expert_submit_both_schedules
```

The deterministic fixture uses seed 3501635, hidden size 16, vocabulary 32,
two layers (GatedDeltaNet/GQA), four experts/layer, top-k three, intermediate
size 16, FP8 weights with BF16 scales and Mixed F32 accumulation. Host budget
is 2 MiB including 256 KiB each for metadata and requested stack; expert cache
is one entry / 64 KiB with 4 KiB scratch. Each selector exercises both schedules.
The error boundary is the first layer-1 routed-down submission before consumer
proof. Unknown is induced by discarding proof **after successful real compute
wait and upload synchronization**, not by a wedged driver or hardware failure.

There is no full-model numerical oracle, 35B fault guarantee, post-Unknown content
equivalence, runtime five-bucket broker ledger, integrated HTTP fault test,
disjoint physical retained-byte total or Unknown release certificate. Unknown
cache bytes are a subset of charged bytes; the separate workspace observation
is not an additional disjoint pool. Independent host cleanup does not release
GPU quarantine.

Optional custodian-only originals remain under
`target/validation/mixed-component-fault-current-20261015-01/`: `gpu-run/FINAL.json`,
`gpu-reports/mixed_clean.jsonl`, `gpu-reports/mixed_unknown.jsonl`, and
`actualreports/final-handoff.json`. The last has SHA-256
`8f653a69707d057160118f536e77aa061209eb0d3914222223d7228d2be5abda`.
These original paths are provenance, not dependencies of public checking or
rendering. The archive guide describes optional read-only private-source auditing;
it does not authorize native test execution or replay of a consumed generation.

## Historical exclusions and provenance

These sources preserve negative evidence and unmatched supplements without
turning artifact chronology into the main scientific argument.

| Record | Locator / exclusion |
|---|---|
| Two stopped transfer-policy pilots | [Attempt 01](experiments/storage_demand/paired_read_v1/http-transfer-v1/pilot-attempt-01.json), [attempt 02](experiments/storage_demand/paired_read_v1/http-transfer-v1/pilot-attempt-02.json): each zero starts, 96 requests / 40 measured QA unissued; neither contributes timing observations to the completed third pilot |
| Rejected retained-upload alternative | [Original negative timing and blocked profiling](experiments/storage_demand/paired_read_v1/engineering-20261016-v1/performance-results.json), [later partial profiles](experiments/storage_demand/paired_read_v1/engineering-20261016-v1/profile-results.json): keep negative outcomes; reproduce only from pinned historical Git, not cleaned production |
| Grouped stage/profile failures | [Whole-test stage](experiments/storage_demand/paired_read_v1/engineering-20261016-v1/grouped-performance-results.json), [profile history and separate qualified pairs](experiments/storage_demand/paired_read_v1/engineering-20261016-v1/grouped-profile-results.json): failed roots remain failed; later approved captures are not retries or healed old captures |
| Two stopped paired studies | [Historical failures](experiments/storage_demand/paired_read_v1/publication_results_v1/historical-failures.json): each one attempted / zero completed lifecycles, 120 HTTP and 50 QA slots unissued, five null pairs; neither contributes to paired-study |
| Earlier pressure attempts | [Pressure history](experiments/storage_demand/pressure_fault_v1/publication_pressure_v1/history.json): one deadline STOP with 3 successful / 1 censored / 8 unissued requests; a separate source-admission rejection with zero starts; neither contributes to cache-pressure |
| Rejected/blocked profile captures | [Profile history](experiments/storage_demand/profile_overlap_v1/publication_v1/history.json): original acceptance is unchanged; retrospective correction is not a newly accepted capture; only two fresh scoped arms qualify |
| Disabled-pipeline storage observation | [Historical analysis](experiments/storage_demand/analysis/README.md), original `runs/storage-demand-tp1-middle-20261013-01/` and `-02/`: prelaunch failure versus one DEBUG four-request run; not the enabled control |
| Earlier numerical/wrapper failures | External `target/validation/storage-middle64-current-quality-20261014-01/` through `-04/` and `storage-observability-cli-quality-20261014-01/`: driver failure, numerical passes with original idle-wrapper failures, or a different-source cold-owner acceptance; later idle does not repair original failures |
| Earlier strict W8A16 oracle | External `target/validation/w8a16-full40-independent-20261002/`: logits/routes/states fail; argmax agreement does not waive failure |
| Resident TP2 supplement | [Original results](experiments/tp2_resident/RESULTS.md), [baseline failures](experiments/tp2_resident/RESULTS-baseline-delta.md), [quality selection](experiments/tp2_resident/model-runtime-quality.json), [crossblocks](experiments/tp2_resident/runs/route-batch-vllm-crossblocks-20261012-01/crossblocks-results.md): different topology and unmatched vLLM controls, not a fair performance ranking |
| Earlier TP2 full profiling capture | [Original audit](experiments/tp2_resident/runs/tp2-official-full-profile-20261010-001/profile-audit-v2.json): incomplete collection and missing API/activity relations; full trace rejected, no performance acceptance |
| Startup metadata supplement | External `target/validation/startup-metadata-owner-20261013-01/REPORT.md`; separate unpaired startup/PSS observations and out-of-window observer costs, not a causal latency or heap-release result |
| Unexecuted baseline pilot | [Independent protocol](experiments/storage_demand/newversionedprotocol/protocol.json): five lifecycles / 60 requests / 25 QA planned, not measured |

### Unmatched resident-system observations

The [original crossblock results](experiments/tp2_resident/runs/route-batch-vllm-crossblocks-20261012-01/crossblocks-results.md)
preserve two resident TP2 windows with fresh owners, three excluded QA requests
and five measured requests per window. These values were previously shown in
the manuscript's resident supplement; they are retained here as ground-truth
historical observations, **not an offload comparison or comparative speedup**.

| Historical window | Output tok/s | Mean client TTFT ms | Mean client E2E ms | Startup s | Sampled GPU GiB/card | Sampled PSS GiB |
|---|---:|---:|---:|---:|---:|---:|
| Ferrule 1 | 14.317 | 176.248 | 523.399 | 349.585 | 21.04 | 61.05 |
| vLLM 1 | 10.376 | 132.676 | 694.228 | 66.239 | 19.28 | 7.71 |
| vLLM 2 | 10.330 | 134.651 | 699.213 | 62.814 | 19.28 | 7.32 |
| Ferrule 2 | 14.804 | 171.333 | 515.106 | 366.133 | 21.04 | 60.14 |

Both use PP1/TP2, but Ferrule's qualified Mixed GemmEx/routing execution and
vLLM's BF16/FP8 Marlin execution do not match arithmetic. A 22 GiB per-device
policy and utilization fraction 0.8 do not match total working-set budgets.
The recorded vLLM version is 0.28.0, CUDA 12.9, Torch 2.13.0. Different metrics
favor different deployments; none establishes a causal winner or SOTA ranking.
They do not qualify an offload combination, another precision, or DeepSeek-V4.
The unpaired startup/metadata observations remain separate as well.

## Primary-source and license audit for positioning

Related-work inspection used already downloaded local primary sources, not a
new live-web snapshot. The local, ignored research index is
`target/research/offload-capabilities/capability-matrix.csv`; its
`source-integrity.json` binds downloaded Git blobs. These local research caches
are **not promised as public artifact payloads**. The bibliography links the
upstream pinned source, not an unavailable local cache.

The added `sglang2026expertpack` citation points to the official SGLang source
at `5ef8319f16290d054295c1499bf82e712a9474c7`, inspected 2026-10-18. Its loader,
runtime store and constraints are respectively:

- `python/sglang/srt/model_loader/expert_pack_loader.py`;
- runtime store: `python/sglang/srt/layers/moe/expert_pack.py`;
- preparation: `python/sglang/srt/model_loader/expert_pack_runtime.py`;
- `python/sglang/srt/model_loader/expert_pack_config.py` and
  `python/sglang/srt/arg_groups/expert_pack_hook.py`.

The loader/store show runtime demand storage reads and a GPU expert cache;
preparation is not the only storage operation. The inspected path accepts
DeepSeek-V4 Flash or the restricted text-only Kimi-K3 configuration, not Qwen3.5;
unit TP/DP/EP and disabled CUDA graphs/shared fusion are explicit checks. This
is not a universal statement about every SGLang adapter/version. Release
inclusion, actual execution and relative quality/performance were not tested.
The new bibliography entry is a source snapshot, not an invented paper,
conference, byline, release date or benchmark. Relevant files carry Apache-2.0
SPDX headers; this update cites their behavior and copies no upstream code.

The capability matrix separately records inspected vLLM 0.28.0+cu129 CPU-weight
offload/UVA paths as host-resident mechanisms, not equivalent disk-weight
demand. KV offload is distinct; arbitrary plugins are not ruled out. FlexGen
and ZeRO-Inference already explicitly include disk/NVMe weight streaming and
are acknowledged as such. No Ferrule-equivalent or absent custody guarantee
is inferred without an exhaustive audit. A full third-party distribution and
model/toolchain licensing audit remains separate from this new citation check.

## Documentation validation

Lead/artifact integration checks completed without a GPU query, release build,
server launch, profiler export or paper build:

- Both newer curated manifests pass SHA-256 and byte-count checks: 10 engineering
  members and eight HTTP members, with no duplicate member paths.
- All 17 cached primary-source blobs pass SHA-256 and Git-blob checks. Five
  relevant SGLang files additionally match the pinned repository tree and decoded
  local source bytes and contain Apache-2.0 SPDX headers.
- The completed pilot's 96 request qualifications, 40 measured requests and
  320 measured tokens reconcile. Client timestamp arithmetic, all eight arm
  TTFT/E2E medians and phase-width/goodput values, and the two paired ratios per
  schedule reproduce from the curated records. This is internal-consistency
  verification, not reauthentication or reconstruction of unpublished raw SSE.
- All assigned guide/ledger/checklist local links resolve. Manuscript citation
  keys and label references (including labels generated by the figure macro)
  resolve at the inspected integration point. `main.tex` differs only inside
  the abstract; title, preamble and input order are unchanged. Assigned lead
  prose contains no project paths, temporary artifact labels or source hashes.
- The scoped Git whitespace check passes. Final LaTeX warnings/boxes, rendered
  figure/table review, grayscale review and source/PDF correspondence are not
  established by these checks and remain unchecked.


Previously recorded focused CPU validation of the public rendering and platform
supplement (not rerun as part of this lead/artifact integration):

- Paired archive checker and **29** standard-library archive tests pass on
  Python 3.10.9; the immutable 623-file manifest remains unchanged.
- **12** public-render tests pass with the pinned Python 3.12.4 environment,
  including detached public-only figure generation and byte comparisons.
- **16** legacy plot tests pass on Python 3.10.9 against retained local evidence;
  these compatibility tests are not a dependency of the public rendering path.
- **29** platform parser/binding and receipt tests pass without live collection.
- Pressure, scoped-overlap and mixed-fault checkers pass; the fault check includes
  its companion figure hashes without rendering that figure.

That is **86 previously recorded focused tests**, not additional experimental
replications. The present integration uses bounded curated-manifest, source-blob,
citation/reference and documentation checks; it launches no GPU work or build.
Final LaTeX integration and visual review remain separate checklist gates. Earlier
qualification/fault/overlap/pressure and HTTP test coverage remains recorded in
their archive documentation. No GPU query, native build, new platform collection
or paper build is part of these checks. Figure rendering here reads public data
and writes only temporary test outputs, never the sealed evidence.
