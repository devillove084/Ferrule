# Qualification v1: same-source candidate quality, not CLI serving

This is a **CPU-verifiable receipt archive of nine historical real-GPU factory/test
passes**, not a new experiment and not a same-binary CLI-serving qualifier.
The sole source cut is `033519c3def098ce417e4b6ff0f141a9f30bddb1`.
Nothing from an older qualification is pooled into the nine-gate denominator.

Original evidence: `target/validation/expert-read-current-quality-20261014-02/`.
Its 1,772 sealed files plus the seal manifest itself were hash-checked and verified
read-only during curation. None was changed. Existing production, manuscript,
paired protocols, models, history, and independent metadata/quality evidence are
outside this directory's ownership. No GPU work, Git staging, commit, or push is
performed by either script here.

## Offline reproduction

From the repository root, with Python 3.10+ and **no third-party packages**:

```sh
python3 -B papers/arxiv/experiments/storage_demand/qualification_v1/offline_check.py
python3 -B papers/arxiv/experiments/storage_demand/qualification_v1/test_offline_check.py
```

The checker also accepts `--archive /absolute/path/to/qualification_v1`. The
complete directory can be relocated and checked without the repository, GPU,
CUDA, libraries, model weights, network, or owner-local original artifacts.
It never executes a process, queries current GPU state, or acquires a lock.
`offline-check.json` and `tests.stdout`/`tests.stderr` record the actual CPU checks.
The exact publication candidate file inventory and byte sizes are in
`STAGE_FILES.tsv`; this is an inventory, **not Git staging**.

To reproduce curation on the original machine, after creating the same empty
archive subdirectories, run:

```sh
python3 -B papers/arxiv/experiments/storage_demand/qualification_v1/curate_archive.py
```

The curator first hashes **all** original sealed inputs and checks their read-only
modes, not just selected records. It writes only within this directory. Archive
parents must already exist: `receiptarchive/`, `receiptarchive/references/`,
`receiptarchive/proof-source/`, `receiptarchive/gates/<each gate>/`, and the two
full40 gates' `numeric-report/` subdirectories. It accepts an already matching
archive, refuses unrelated files, and does not run build/qualification adapters.
Compression is gzip level 9 with timestamp zero; hashes bind both compressed
storage and the exact uncompressed bytes, so validation does not depend on gzip
byte identity across Python/zlib versions.

## Identity and external payloads

| Object | SHA256 |
|---|---|
| CLI candidate | `de5159951fe724e97e35c0fc86367642d33735994328eee1a5e604cf9d2b037b` |
| Composition test executable | `8810c3418f4b39b4c9203003b91e53a6106a833624db4d13a516fdfd63453d6a` |
| Constructor test executable | `cb9b8c998449b8f73a6a6919be19bcd8e1d743711a1c06d36e5332af57db4a23` |
| Original committed-source tar | `ac072e0d1bac7efcc85b7f26ce68723f412b0e22b5ee868104b1f6073dd2a82f` |
| Absolute-path source-pin digest | `a4dceb084e21b80f6b45ff7ac980c6fa80ac8f17e902fd28c65f3d5bc7bb0efa` |
| Relative source-member pin digest | `bde8b14396855df4cf832cb89d01250f2b4e162ef7ed9b3abc93d2c4df975fa7` |
| Original evidence seal | `0b6311f25951d4bb89724cf53caa57d47475dab6c46b70025d37f451c30b7ceb` |
| Original CPU receipt audit | `5a662435f90c40e733ffa39171340e9bee6b13d046ca6eb2bd800d8d46cb53b1` |
| Sealed nine-gate plan | `f83ab450578b574c25fa48d4039f66f6df1a91725d586e1c029b7f0b1f7a066d` |

The external CLI, composition, constructor, and source tar are respectively:

```text
/home/admin/luhuanbing.lhb/Ferrule/target/validation/expert-read-current-quality-20261014-02/staged/cli
/home/admin/luhuanbing.lhb/Ferrule/target/validation/expert-read-current-quality-20261014-02/staged/composition
/home/admin/luhuanbing.lhb/Ferrule/target/validation/expert-read-current-quality-20261014-02/staged/constructor
/home/admin/luhuanbing.lhb/Ferrule/target/validation/expert-read-current-quality-20261014-02/source.tar
```

`manifest.json` records these paths, hashes, and sizes. Library/native archive
paths and hashes remain in `runtime-library-pins.json` and
`native-archive-pins.json`; dependency/toolchain/header paths and hashes remain in
`compiled-dependency-pins.json`, `external-input-pins.json`, and `toolchain.json`.
Model location: `/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8`. Model metadata content hashes
and shard identity observations are retained in `reference-pins.json` and
`model-shard-identities.json`. Shard stat/header identities are not invented
whole-model content hashes. Oracle tensor payload paths/content hashes are in
`reference-pins.json` and the curation manifest's external-artifact list.
**The relocated offline check does not read or rehash these external payloads.**

No CLI/test executables, `.so`, native `.a`, model/tensor payloads, source tar,
build objects, or complete 1,484-member source tree are archived. Only six small
numerical/counter-proof Rust files are retained, individually pinned as members
of the original source freeze. All other source members are represented by the
exact source/member pin manifests, not copied.

### Recover the exact committed source, not today's worktree

For a source inspection independent of the six proof copies:

```sh
git --no-pager show 033519c3def098ce417e4b6ff0f141a9f30bddb1:crates/ferrule-model/src/transformer/expert_reads.rs
```

To recreate the original tar in a **new** reproduction directory (do not replace
the sealed original), run from the repository root:

```sh
mkdir -p target/reproduction/qualification_v1
git --no-pager archive --format=tar --output=target/reproduction/qualification_v1/source.tar 033519c3def098ce417e4b6ff0f141a9f30bddb1 -- Cargo.toml Cargo.lock rust-toolchain.toml README.md .cargo crates scripts papers/arxiv/experiments/contract.py papers/arxiv/experiments/launch_guard.py papers/arxiv/experiments/runner.py papers/arxiv/experiments/m7.json papers/arxiv/experiments/m7-device-exclusive.json papers/arxiv/experiments/storage_demand/case-middle.json
shasum -a 256 target/reproduction/qualification_v1/source.tar
```

Expected tar hash is the source-archive hash above. `source-freeze.json` preserves
the exact original argv, all member hashes, and both digest inputs. Source-pin
digests are SHA256 of `json.dumps(pins, sort_keys=True, allow_nan=False).encode()`;
the absolute-path digest intentionally includes original paths.

### Build reproduction is distinct from receipt verification

The original `build-<role>-command.json` records exact argv, allowlisted environment,
cwd, and deadline; `build-<role>-receipt.json` and stdout/stderr retain the actual
build outcome. The three Cargo invocations, from recovered source and with the
recorded build environment, are:

```sh
cargo build --locked --offline --release --no-default-features -p ferrule-cli --features cuda,nvtx-profile --message-format=json
cargo test --locked --offline --release --no-default-features -p ferrule-model --features cuda,ferrule-common/nvtx-profile --test expert_read_constructor --no-run --message-format=json
cargo test --locked --offline --release --no-default-features -p ferrule-composition --features cuda,ferrule-common/nvtx-profile --lib full40 --no-run --message-format=json
```

Use a **new** Cargo target/output directory, not the sealed evidence directory.
The original setup is Linux x86-64, Rust `nightly-2026-08-01`, CUDA 12.4.99, SM86,
CUTLASS `e05f953a5b3d38adc240df2ff928e0421c2abba3`, and official installed NVTX3
headers. The exact public build environment is in `build-environment.json`.
Common NVTX support is compiled; qualification runtime `FERRULE_PROFILE_NVTX=0`.
Dependencies/vendor/native/runtime inputs must match their pinned metadata;
`--offline` requires those dependencies already installed. Relocating build paths
or changing tools may change executable bytes: same source is not an assertion
of bitwise reproducible binaries or mathematical binary equivalence.

This is **not authorization for a GPU rerun**. Per-gate authenticated-launch
records preserve historical argv/environment, while the sealed plan identifies
selectors, admission, idle windows, locks, deadlines and zero-retry policy. A new
real-GPU reproduction needs a fresh reviewed immutable build and maintained
launch/custody authorization, not direct test execution or resumption of a stopped
HTTP study.

## Actual acceptance denominators

| Gate | Main metrics | Main logit values | Actual fingerprints | Positive prediction calls / sampled operations |
|---|---:|---:|---:|---:|
| 01-constructor | 0 | 128 finite tiny values | 0 | 4 cells |
| 02-blocking-full40 | 804 | 1,986,560 | 0 | 4 / 128 |
| 03-lookahead-full40 | 804 | 1,986,560 | 0 | 4 / 128 |
| 04-blocking-fresh | 5,427 | 9,187,840 | 64 | 64 / 963 |
| 05-lookahead-fresh | 5,427 | 9,187,840 | 64 | 64 / 963 |
| 06-blocking-ordinary | 27 | 9,187,840 | 64 | 64 / 66 |
| 07-lookahead-ordinary | 27 | 9,187,840 | 64 | 64 / 66 |
| 08-blocking-cold | 1,809 | 3,228,160 | 28 | 28 / 369 |
| 09-lookahead-cold | 1,809 | 3,228,160 | 28 | 28 / 369 |

All nine were dispatched exactly once, passed one actual selected test, and had
complete accepted reports. No ignored-test summary, missing report, retry, probe,
warmup, or profiling capture is accepted. Constructor coverage is four
boundary/schedule cells, each with 32 finite logit values and six verified reads;
it is **not** a full-model numerical oracle.

For **each** fresh or ordinary schedule, keep streams separate:

- Common history: 27 prediction calls / 9,187,840 logit values.
- Independent free feedback: 27 / 9,187,840.
- Reset replay: 9 / 3,228,160.
- Same-history fork replay: 1 / 248,320; not divergent KV COW proof.

For **each** cold schedule: common history, free feedback and reset each cover
9 / 3,228,160; fork covers 1 / 248,320. Do not sum these semantically different
streams into a single quality denominator. Independent metadata and prior
quality/reference acceptance, including separately maintained 31-ID and HF-scale
ID checks, are not replaced, enlarged, or requalified here. This checker does not
reproduce those independent gates. Original numeric/HF scale and token-reference
pins are preserved, not recalibrated or relabeled.

Ordinary execution matches **all 64** same-schedule diagnostic actual F32-word
SHA256 fingerprints, shapes, and greedy IDs. Both ordinary and cold runs bind the
current fresh primary's config, oracle and hash, not an older PASS. Cold's 28
fingerprints are its own coverage, not a claimed 64-call ordinary comparison.
The independent reference comparator remains `0.00025 + 0.012 * abs(reference)`,
with exact route/greedy ID checks and lower-ID tie policies.

## Effective modes, counters, and resource limits

Both schedules record positive worker reads and their actual effective modes.
At the **last sampled owner-lifetime endpoint**:

| Gate kind | Verified / worker-started reads, either mode | Blocking prefetched/speculative | Lookahead prefetched/speculative |
|---|---:|---:|---:|
| Full40 | 2,089 | 0 | 1,929 |
| Fresh | 25,116 | 0 | 22,556 |
| Ordinary | 25,116 | 0 | 22,556 |
| Cold | 11,354 | 0 | 10,234 |

These are sampled owner-lifetime counters at different run times, **not additive
invocation counts, counterfactual paired performance, GPU overlap, or speedup**.
`reader_counter_scope` explicitly identifies shared checkpoint reader intervals
from submit through take, not certified per-ticket syscall or physical NAS bytes.
Actual worker/IO/source-verified receipt timestamps are host-monotonic offsets,
not CUDA event timing. Numerical metrics in exact stderr and the individually
pinned numerical source/counter code provide inspectable producer evidence;
no log event subscriber or observational snapshot supplies new resource authority.

Frozen composition capacities: host 64 MiB including metadata 8 MiB and worker
stack 2 MiB; device cache 64 entries / 320 MiB including scratch 8 MiB; device
budget 7 GiB; workspace 2 GiB; KV 16 MiB; context 128; batch/prefill 8; sequences 2;
decode cohort target/deferrals 1/0. Observed allocator peak is 4,106,223,616 bytes;
logical host charge peak is 16,777,984 bytes. Both endpoints and component peaks
are checked against declared bounds. Host charge is
`logical_host_reservation_not_physical_rss`: **neither observed physical RSS nor a
physical host-memory certificate**. Logical state-after-shutdown zero also is not
a CUDA release certificate.

All processes completed naturally without cleanup signals. Authenticated launch
IDs, helper/coordinator identity, PID/birth/parent identity, maintained subreaper
custody, ECHILD, owner absence and three consecutive all-eight idle samples are
cross-checked. All eight UUID local locks were released and nonblockingly
reacquired. Idle and lock observations are historical and local, not current
machine state or global GPU reservations. **Physical CUDA release remains
unobserved**. Prior CUDA driver flakiness still has no established root cause.

## What is copied and what is projected

`manifest.json` is the copy-input inventory. Each entry records original
repository-relative path/content hash/byte length, destination, encoding,
stored hash/bytes, uncompressed hash/bytes and transformation. Exact raw textual
copies exceeding 64 KiB use `.gz`; JSON/log bytes are not reserialized, trimmed,
pooled or sanitized except for the explicitly labeled worker projection.

The archive contains original SUMMARY, FINAL, evidence seal and CPU audit;
per-gate typed reports and producer gate-result; authenticated launch and custody;
exact selector list and test stdout/stderr (including empty stderr); public
runtime environment; actual PID/birth runtime mapping snapshots; three idle
receipts; final idle/UUID-lock evidence; source/config/reference/build/ELF/native
link metadata; original reference JSON with only public canary prompts.

`worker-launch.public.json` removes **only** `worker_token`. Its original sealed
content hash is retained, but cannot be reconstructed offline without the omitted
credential. Raw dispatches are excluded because argv contains worker admission
credentials. Original dispatch/worker record hashes remain in the unchanged
original evidence manifest. Public projections bind gate, plan, coordinator,
allowlisted environment and locks; offline identity checks compare the exact
launch/supervisor/custody records. They do not replay or freshly authenticate an
old process handshake. No raw environment dump or launch credential is published.

## Independent HTTP failure pointer — not part of these nine gates

See `../paired_read_v1/RESULT_V2.md` and
`../paired_read_v1/artifacts-current-20261015-01/closeout.json`, owned separately.
The same CLI candidate's first planned HTTP arm failed **before Ready**, with
`prepared source profile changed`, through the prepared TP2-cache diagnostic
accessor on a TP1 numeric request. No HTTP request was issued and no CLI-serving
PASS is established. The factory/test PASS remains valid but does not exercise
that accessor. No raw paired-study artifacts are copied here; no protocol, case,
production fix, retry, replacement arm or paired-performance claim is made.
