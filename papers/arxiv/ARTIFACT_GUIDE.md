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
| **factory-qualification**: nine gates on the serving source | [Build and quality projection](experiments/storage_demand/paired_read_v1/publication_results_v1/design/source-and-build.json) | [Paired checker](#public-cpu-checks), projection only; [external originals](#external-current-qualification) | Qualification table, not a performance figure; complete current raw qualification is not a portable public archive |
| **earlier-factory-qualification**: nine gates on a different source | [Qualification archive](experiments/storage_demand/qualification_v1/README.md) | [Qualification checker](#public-cpu-checks) | Historical qualification only; not another nine serving-source repetitions |
| **paired-study**: five completed pairs, ten lifecycles, 120 completed requests; 50 measured QA | [Paired receipts and results](experiments/storage_demand/paired_read_v1/publication_results_v1/README.md) | [Paired checker](#public-cpu-checks) | [Latency](figures/paired-read-latency.pdf), [read wait](figures/paired-read-read-wait.pdf), [accounting](figures/paired-read-accounting.pdf); [public figure regeneration](#paired-public-figures) |
| **cache-pressure**: three completed blocking lifecycles; 12 completed requests | [Pressure projection](experiments/storage_demand/pressure_fault_v1/publication_pressure_v1/README.md) | [Pressure verifier](#public-cpu-checks) | [Cache budgets](figures/cache-pressure-budget.pdf); one fixed-order coupled entry/byte sweep, not a capacity frontier or repeated timing study |
| **scoped-overlap**: two accepted profiled QA arms | [Interval archive](experiments/storage_demand/profile_overlap_v1/publication_v1/README.md) | [Overlap checker](#public-cpu-checks) | [Read/compute overlap](figures/read-compute-overlap.pdf); scoped acceptance true, full-trace acceptance false, dropped-event count null |
| **mixed-fault**: two selectors / four actual component cells pass | [Portable actual-fault archive](experiments/storage_demand/pressure_fault_v1/publication_fault_v1/README.md), [typed evidence](experiments/storage_demand/pressure_fault_v1/publication_fault_v1/SCHEMA.md) | [Fault checker and 27 tests](#public-cpu-checks), no private inputs | [Fault custody](figures/fault-custody-contract.pdf); tiny Mixed CUDA fixture, not the serving executable, a 35B fault experiment or an integrated broker ledger |
| **http-cancel**: one lifecycle, two requests; one controlled censor and one exact recovery pass | [Public receipt](experiments/storage_demand/http_cancel_v1/receipts/result.json), [results](experiments/storage_demand/http_cancel_v1/RESULTS.md) | [HTTP receipt tests](#http-receipt-check) in a repository checkout | Cancellation/recovery table, no performance figure; cancellation is not an eight-token QA success or proof that a synchronous read was interrupted |
| **platform-inventory (post hoc)**: CPU/kernel-visible memory recorded later on the paired-study boot | [Inventory receipt and scope](experiments/platform_inventory_v1/README.md) | [Offline inventory tests](#platform-inventory-offline-check) | Methods metadata, not a new performance or capacity observation; [same-boot caveats](#post-hoc-platform-inventory) apply only to paired-study |

Detailed scalar tables live in those result documents, not in parallel copies
here. Requests, paired blocks, component cells, and distinct source cuts must not
be pooled into a larger sample size.

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

The serving executable used for paired-study, cache-pressure, scoped-overlap,
and http-cancel stays bound to source
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

The following are compact integrity entry points, not a complete Git history.
Each manifest supplies member-level hashes and the original/projection distinction.

| Public input identity | Publication commit, where recorded | SHA-256 of the named manifest or receipt |
|---|---|---|
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

The full-model studies use Qwen3.5-35B-A3B-FP8, Mixed precision, PP1/TP1/EP1,
lazy experts and resident KV. The common middle configuration is context 128,
batch/prefill 8, two sequences, decode cohort target 1 / maximum deferrals 0;
device policy 7 GiB, workspace 2 GiB, margin 512 MiB, KV 16 MiB. The host read
reservation is 64 MiB including 8 MiB metadata and a requested 2 MiB worker
stack; one worker, queue depth one and two payload slots. The expert-cache
policy is 64 entries / 320 MiB including 8 MiB scratch. These are logical
accounting/configuration scopes, not a total physical-memory partition.
Cache-pressure couples caps at 8/64, 64/320 and 128/640 entries/MiB; all use
blocking. There is no 256-entry observation.

Both paired arms enable the real read pipeline; only `--expert-read-schedule`
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

## External current qualification

The current nine-gate raw evidence is local/external, under
`target/validation/fixesmetadataCLIcurrent-quality-20261015-01/`.
`FINAL.json`, `qualification-plan-v2.json`, `gates/*/quality.json`, the full40
`gates/*/numeric-report/full40-bf16-profile.json` reports, and their supervisor
records are **not a complete portable archive**. The public
[source/build projection](experiments/storage_demand/paired_read_v1/publication_results_v1/design/source-and-build.json)
records all nine report/supervisor hashes, executable identities, acceptance
rows and reference pins. Its check validates that projection, not the missing
raw qualification corpus. The older public archive does not fill this gap.

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

## Documentation validation

Focused CPU validation of the current public rendering and platform supplement:

- Paired archive checker and **29** standard-library archive tests pass on
  Python 3.10.9; the immutable 623-file manifest remains unchanged.
- **12** public-render tests pass with the pinned Python 3.12.4 environment,
  including detached public-only figure generation and byte comparisons.
- **16** legacy plot tests pass on Python 3.10.9 against retained local evidence;
  these compatibility tests are not a dependency of the public rendering path.
- **29** platform parser/binding and receipt tests pass without live collection.
- Pressure, scoped-overlap and mixed-fault checkers pass; the fault check includes
  its companion figure hashes without rendering that figure.

That is **86 focused tests**, not additional experimental replications. Earlier
qualification/fault/overlap/pressure and HTTP test coverage remains recorded in
their archive documentation. No GPU query, native build, new platform collection
or paper build is part of these checks. Figure rendering here reads public data
and writes only temporary test outputs, never the sealed evidence.
