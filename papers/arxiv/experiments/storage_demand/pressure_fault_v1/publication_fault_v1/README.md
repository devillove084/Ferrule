# Actual Mixed fault component: portable publication v1

**Two actual CUDA selectors, four cells, four PASS.** Each selector ran Blocking
then Lookahead. This is a tiny **component contract**, not the production CLI,
not integrated broker/E5 evidence, not a 35B model fault experiment, and not a
physical driver/event-proof-loss experiment. Publication uses already completed
runs: **no new models, GPU probes, executions, retries, or package installations**.

## Minimal offline reproduction

Python **3.10+**, standard library only, no network, CUDA, model, Git checkout,
`target/` directory, third-party Python package, or private input is needed.
From this directory (or an isolated copy of this directory):

```sh
python3 -S -B check.py
python3 -S -B -m unittest discover -s . -p 'test_*.py' -v
```

From the repository root, optionally verify the companion source/PDF/PNG too:

```sh
python3 -S -B papers/arxiv/experiments/storage_demand/pressure_fault_v1/publication_fault_v1/check.py --figures papers/arxiv/figures
```

The checker verifies actual producer bytes, source identities, ordered records,
one fault per cell, frontier/custody invariants, independent host cleanup,
free/charged arithmetic, evidence types, and the figure's data and text layout.
`postfailure.json` is regenerated **in memory** and compared to the published
projection; it is not accepted as an independent measurement.
The tests generate corrupt metrics, source identities, duplicate faults,
frontiers, false quiescence, false post-Unknown content equality, refunded
Unknown custody, and even **rehashed** false projections. An isolated-copy
subprocess uses `-S` and a minimal environment.

### Companion figure

- `papers/arxiv/figures/fault-custody-contract.py`: deterministic renderer source.
- `papers/arxiv/figures/fault-custody-contract.pdf`: vector figure with embedded
  TrueType fonts, suitable for manuscript inclusion by its owner.
- `papers/arxiv/figures/fault-custody-contract.png`: 2052 × 1728 preview.
- `papers/arxiv/figures/fault-custody-caption.md`: caption and claim boundaries.
- `figure.json`: exact plot input, source/image hashes, renderer/font versions,
  measured text bounds, and zero-overlap layout audit.

Rendering, unlike data verification, uses an **already installed** Matplotlib
stack. The actual rendering environment was Python 3.12.4, Matplotlib 3.8.4,
NumPy 1.26.4, Pillow 10.3.0, FreeType 2.12.1, zlib 1.2.13, DejaVu Sans, Agg.
Do not install anything merely to check the archive. With a Python interpreter
that already provides those versions, from the repository root:

```sh
python3 -B papers/arxiv/figures/fault-custody-contract.py
python3 -B papers/arxiv/figures/fault-custody-contract.py --check
```

For a detached bundle, supply `--archive /path/to/publication_fault_v1` and
`--output-dir /path/to/companion-figures`. These are user-chosen local paths;
the renderer does not discover the original private run. `--check` re-renders
in memory and compares image/metadata bytes; Matplotlib may populate its local
ignored font cache. Different rendering-library/font versions may give different
image bytes without changing the checked data. Dates are omitted from PDF
metadata, and PNG metadata is fixed.

## What actually ran

Fixture: known seed **3501635**, hidden/intermediate width **16**, vocabulary 32,
two layers (**GatedDeltaNet, GQA**), **four experts, top-3**; numeric FP8 weights
with BF16 scales, BF16 activations/history/KV and F32 SSM
(`Bf16ModelMixedF32Accumulate`). Cache budget **65,536 B**, one entry, persistent
numeric scratch reservation **4,096 B**. No weights or model binaries are
published. This fixture is not a numerical-quality reference.

Tx1 publishes token `[1]`: committed frontier **1**, GDN position 1, generation
0, committed KV page at slot **0**. Tx2 prepares continuation `[2,3]` and hits a
one-shot scheduled error **after the real first layer-1 expert-0 routed-down
submission**. The serialized boundary observes two additional compute launches
(372 → 374), and actual compute-event completion plus upload synchronization
succeed. The fault is scheduled by private test-only source, not a hardware
failure.

| Post-cleanup observation (both schedules) | Clean | Synthetic Unknown |
|---|---:|---:|
| Committed frontier / KV slot | 1 / 0 | 1 / 0 |
| Live state bytes: default + committed + working | 1,024 | 1,536 |
| Active runner / KV transactions | 0 / 0 | 1 / 1 |
| Free KV slots | 31 | 29 |
| Cache free bytes | 61,440 | 58,618 |
| Cache charged bytes | 4,096 | 6,918 |
| Pending upload bytes | 0 | 774 |
| Scratch bytes | 4,096 | 6,144 |
| Unknown **subset of charged** | 0 | 6,918 |
| Current host charge / host Unknown bytes | 0 / 0 | 0 / 0 |
| Post-fault KV / convolution / SSM content equality | Test asserts bit-exact | **UNOBSERVED** |

### Clean

The real owner completion proof is retained. Abort returns the **512 B working
state and two transient KV slots**, not all state memory. Default plus committed
state remain 1,024 B; persistent numeric scratch remains 4,096 B. The test
compares committed BF16 KV, BF16 convolution history, and F32 SSM bits to their
pre-fault values. A healthy **in-test tx3** executes the same continuation and
is then aborted; it is **not** another selector run, model run, or publication.

Only after separate explicit committed-page/state release, runner/image drop,
and allocator retirement trim does the fixture's allocator-requested-byte
observation return **baseline 0 → final 0**. This is **fixture-local**, not a
passive snapshot of immediate zero retention and not device-wide free memory
or a global CUDA release certificate.

### Unknown

The actual consumer compute-event wait and upload synchronization first succeed;
the test then **deliberately discards their owner proof**. Later, **two independent
compute events and two independent upload events also synchronize successfully**.
Each of the two subsequent abort attempts still returns typed
`CompletionUnknown`. Owner holds, working state, KV custody, and charged cache
remain retained. Reset, new sequence admission, KV release, publish and shutdown
are rejected. The runner, state and device are intentionally retained until
process exit; no in-process refund or allocator-zero claim is made.

**Post-Unknown KV / convolution / SSM CONTENTS ARE UNOBSERVED.** Only committed
metadata remains unchanged. Independent waits do not authorize an unsafe device
read or repair the missing owner proof. A source assertion/PASS is not a newly
serialized API-error receipt; see [SCHEMA.md](SCHEMA.md).

### Accounting boundaries

For every cache snapshot, `resident + pending + scratch = charged`, and
`free + charged = 65,536`. `unknown_subset <= charged`; it is hatched **inside**
the charged bar, never added as a third capacity bucket. This fixture contains
**no runtime broker five-bucket ledger**.

The **318,720 B recurrent workspace** is a separate, potentially overlapping
owner view. Cache, workspace and state observations must **not** be summed into
an alleged disjoint physical total. The overlap and exact KV reference counts
are not measured and stay `null` in the projection.

Host reads shut down independently. Each Blocking fault cell records five
completed reads, 3,870 proved bytes, no wasted read; each Lookahead cell records
six completed reads, 4,644 proved bytes, **one wasted/cancelled read = 774 B**.
Host charge, Unknown reads and Unknown bytes are zero after cleanup in all four
cells. This does **not** discharge GPU quarantine. Whole-tx2 CUDA counters are
183 compute launches, 50 H2D copies, 15 artifact uploads / 3,870 bytes per cell;
they are not per-read attribution or timing/overlap measurements.

## Evidence, custody and privacy

`evidence/` contains the two **actual** producer JSONL reports, two harness
stdout files and matching structured stderr files, byte-identical to pinned
originals. Every report is 40 rows: header, two ordered 19-row cells, seal. The
closed vocabulary in `report-contract.json` accepts reviewed enum strings and
unsigned integer/bool/null fields; new arbitrary text is rejected, not silently
redacted into evidence. The original header's `gpu_execution: unobserved` and
cell entry's `running_not_yet_proved` are preserved; observed outcomes and seals
follow them. Preparation documents are not retroactively promoted to execution.

`custody.json` is an allowlisted projection of original ownership/ECHILD/NVML
receipts. It preserves the two process outcomes and eight idle windows: **26
samples total**, including **two utilization-only post-run transients** followed
by three accepted idle samples each. All eight device ordinals are retained;
UUIDs, PID/birth/session values and authentication material are not. ECHILD,
recorded-owner absence, local lock release and NVML idle **are not** a physical
CUDA-release certificate. Public authentication conclusions are pinned audit
assertions, not independently replayable handshakes.

No raw environment, worker token, authentication key, GPU UUID, raw process map,
executable, shared object, native archive, source tar or model file is included.
Literal field names in exclusion documentation/tests are not credentials.
`source-manifest.json` retains relative paths and SHA256 pins for all **1,487**
frozen source files, the nine selected source references, producer receipts,
original input manifests, native/library basenames and digests, and safe toolchain
versions. The original frozen copy was **read and hashed, never altered**.

Hashes detect drift; they are not signatures. An adversary replacing both the
checker and its trust anchors is outside this archive's integrity guarantee.
The portable bundle does not claim to independently authenticate redacted raw
handshakes or to rebuild/run the CUDA component without its external toolchain.

## Immutable identities and optional private audit

| Identity | Pin |
|---|---|
| Component source commit | `79d684f5c79fc6c3961161111ae6efd1a37ea8ea` |
| Archive commit | `bd0673579286e614079ccfe5d972be19f860e9e8` |
| Selected source archive SHA256 | `95423c59156abc391eb0a3d80c3c1b2c368bddb5366931456d0bebf1fdd18983` |
| 1,487-file source inventory digest | `c5f447cd814c8b672c4a6afc380f34a3633478946ee44a68b4143e5c8d238097` |
| Actual executable SHA256 | `e531be47a5113cc53665fdc6576cf389b56c29cb7e16f275efec52b0f79b6903` |
| Actual final-handoff SHA256 | `8f653a69707d057160118f536e77aa061209eb0d3914222223d7228d2be5abda` |

The pinned actual handoff records selected-source equivalence of the two commits;
publication did not run a new Git comparison. The original source build used
nightly 2026-08-01, CUDA 12.4.99, `sm_86`, CUTLASS
`e05f953a5b3d38adc240df2ff928e0421c2abba3`, offline locked Cargo, and common NVTX
support compiled but profiling disabled. `reference/` is a byte-pinned **source
reading subset**, not a standalone build tree. The source commit plus the complete
relative inventory identifies the full build inputs.

For an owner who already has the original validation cut and pinned external
inputs, this optional command only reads/hashes files, never launches a GPU or
model process:

```sh
python3 -S -B papers/arxiv/experiments/storage_demand/pressure_fault_v1/publication_fault_v1/archive.py --source target/validation/mixed-component-fault-current-20261015-01 --verify-source-only
```

Omit `--verify-source-only` only when intentionally regenerating this owned
publication payload. It does not rewrite the 1,487-file frozen source copy.
This optional private-source audit is **not required** for any public check or
figure rendering. No GPU rerun command is authorized by this publication.
