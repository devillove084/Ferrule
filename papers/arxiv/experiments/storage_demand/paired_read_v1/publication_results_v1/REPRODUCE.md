# Reproduction levels and external inputs

## 1. Offline analysis of this experiment (fully portable)

Run `python3 -S -B check.py` and `python3 -S -B test_publication.py` from a copy of
this directory. Only Python 3.10+ and these files are required. `--json` prints the
full reconstructed analysis/token/counter report. The checker makes no network,
NAS, GPU, model, compiler or Git calls. No bootstrap seed or numerical package is
needed because all 5^5 block resamples are enumerated exactly.

`prespecified.py` is the frozen `analysis.py` implementation with its external
`finite`/`require` import replaced by local equivalent helpers. Original
`analysis.py` and `quality.py` hashes are in `provenance.json.gz`. `reconcile.py`
implements portable raw-value checks rather than importing the launching harness.
The new code is post-study curation, not a preregistered additional estimand.

## 2. Owner-side reproduction of the public projections

This needs the private, unchanged original trees at their recorded repository
locations. It is not necessary to run the public checker. Do **not** overwrite this
release. In a separate checkout containing the same private inputs and a fresh
publication destination, use the included curation sources, then:

```sh
python3 -S -B curate.py
python3 -S -B seal.py
python3 -S -B check.py
python3 -S -B test_publication.py
python3 -S -B audit_sources.py
```

The curator is intentionally fixed to this generation and this directory name;
its helpers are not a general-purpose raw-data exporter. The sources read only
local receipts/metadata; they never launch a study or change original artifacts.
`seal.py` creates derived files exclusively; existing differing result bytes are
rejected, not overwritten. `common.seal` hashes only this new publication directory.

## 3. Rebuild the executable (external, not performed here)

The immutable build identity is:

| Artifact | Identity |
|---|---|
| Source commit | `86c58a017d7c4b53490f536a3b1a133f2f7848cc` |
| Original source archive SHA-256 | `e0d160996147820fcc64cfb29070b265b2297a6d9265a21de541ed77e8f2fa65` |
| Source-manifest digest | `10183d246a1278248273e7d3055d1690cb56854cbff30f9f0662f18813dc8188` |
| Archived source members | 1,484; three hash-allowlisted generated Python bytecodes are not compiled inputs |
| Staged CLI SHA-256 | `7cc8e0ff6e84f27090494b8e9e27dc3a388d1d2f3fc34ae39dae416e1e607a01` |
| Constructor gate executable SHA-256 | `693fe2de0983432a12e46d4c079bef370466434ce2ac181bf4af2dffba24ef1f` |
| Composition gate executable SHA-256 | `0a429c47c158bbe25f91730b75b47fd5cb768106f1d63d844da3a5f84fc8677f` |

These are hashes of the **original external artifacts**, not claims that the
artifacts are included. The source digest used original absolute manifest keys;
relocating those keys changes that manifest digest. Preserve the original sealed
manifest when verifying it rather than inventing a new equivalent digest.

The actual source archive command selected the following committed inputs:

```sh
git --no-pager archive --format=tar 86c58a017d7c4b53490f536a3b1a133f2f7848cc -- Cargo.toml Cargo.lock rust-toolchain.toml README.md .cargo crates scripts papers/arxiv/experiments/contract.py papers/arxiv/experiments/launch_guard.py papers/arxiv/experiments/runner.py papers/arxiv/experiments/m7.json papers/arxiv/experiments/m7-device-exclusive.json papers/arxiv/experiments/storage_demand/case-middle.json > source.tar
```

Run the following commands **from that separately provisioned, verified immutable
source tree**, not the current working tree. They are the recorded commands with
only the host-specific cargo executable location replaced by `cargo`:

```sh
cargo build --locked --offline --release --no-default-features -p ferrule-cli --features cuda,ferrule-common/nvtx-profile --message-format=json
cargo test --locked --offline --release --no-default-features -p ferrule-model --features cuda,ferrule-common/nvtx-profile --test expert_read_constructor --no-run --message-format=json
cargo test --locked --offline --release --no-default-features -p ferrule-composition --features cuda,ferrule-common/nvtx-profile --lib full40 --no-run --message-format=json
```

The original bounded build timeout was 1,800 seconds per command. Release profile:
opt-level 3, fat LTO, one codegen unit, stripped symbols; no debug assertions.
The CLI and both quality executables compile NVTX, while runtime profiling is off
in both study arms (`FERRULE_PROFILE_NVTX=0`). No CPU-reference feature is substituted.

External provisioning must supply the exact compiler/runtime/native input pins
listed in `provenance.json.gz` and `design/source-and-build.json`, including:

- Rust toolchain `nightly-2026-08-01`; cargo `1.99.0-nightly` (`7c83d4cc0`),
  rustc `1.99.0-nightly` (`ad3d0bc141a02cf446e384136d250a1f6950fed5`), LLVM 22.1.8.
- CUDA toolkit **12.4.99**, CUDA architecture **sm_86**, pinned host C/C++ toolchain,
  loader, driver and CUDA/BLAS runtimes. Twelve required runtime library hashes
  are listed by basename; no shared libraries are bundled.
- CUTLASS commit `e05f953a5b3d38adc240df2ff928e0421c2abba3` and the pinned official
  NVTX3 headers from Nsight Systems CLI **2025.3.1**. The include root must contain
  `nvtx3/nvToolsExt.h`; no substitute headers or profiling capture are implied.
- The Cargo dependency cache for `--locked --offline`, pinned Rust toolchain files,
  and exact source/native compilation inputs. No network download occurs in this
  bundle's tools. Acquiring external dependencies separately may require network
  access and accepting their licenses.
- Locally provisioned values for `CARGO_HOME`, `RUSTUP_HOME`, `CARGO_TARGET_DIR`,
  `CUDA_HOME`, `NVCC`, `FERRULE_CUTLASS_DIR`, `FERRULE_NVTX_INCLUDE_DIR`, `CC`, `CXX`,
  `CUDAHOSTCXX`, `RUSTC`, `RUSTDOC`, library search paths and temporary directory.
  Host paths and complete inherited environment maps are intentionally not copied.
  Public controls are recorded in `design/source-and-build.json`: build jobs 4,
  incremental 0, auto-install 0, CUDA test oracle 0, workspace `:0:0`, profiling 0.
  Original CPU build steps hid the GPU; no driver initialization is needed to
  compile. Build custody was bounded but **not authenticated GPU-producer custody**.

A successful compile alone does not prove bitwise equality, reproduce the nine
authenticated numerical gates or qualify a new CLI for serving. Original native
snapshots and staged executables must be obtained from the owner if their exact
content hashes are to be checked. This archive does not distribute them.

## 4. Repeat the experiment (separate authorization and new generation)

Repeating the experiment needs the verified Qwen3.5-35B-A3B-FP8 checkpoint/tokenizer,
independent oracle/reference inputs, approved GPU/NFS host, same eight-device
inventory and selected device, maintained authenticated supervisor, nine accepted
numerical producer gates, and all original runtime/build pin checks. Tokenizer
and reference hashes are in `design/oracles.json`; model/weight payloads, private
authentication and mount/network locations are not included.

The maintained runner sources and immutable input hash pointers are recorded in
`provenance.json.gz`. Use a **new reviewed plan, new output destinations, fresh
owner authorization, locks/idle admission and current binding**, rather than
reusing or resuming existing sealed paths. No launch command is provided here
that could accidentally consume the historical generation. This publication is
not GPU authorization. `final_adapter_v3/mapfixed.py` is the historical maintained
entry, not an invitation to replay its consumed `start`/`next` commands.

Configuration is recorded in `design/cases.json` and every arm's configuration:
PP/TP/EP=1, context 128, 64 expert cache entries, 320 MiB device expert budget
including 8 MiB scratch, 7 GiB device budget, 2 GiB workspace, 16 MiB KV, logical
read-host budget 64 MiB including 8 MiB metadata and 2 MiB requested stack.
The remaining payload reservation is 54 MiB (27 MiB packed + 27 MiB artifact).
OS stack rounding/TLS/allocator overhead and total server RSS need extra headroom.
Scheduler cohort target is 1 and maximum deferrals 0. Both arms use identical
logging/observation settings; their overhead is unmeasured, not assumed zero.

Five paired blocks execute P1..P5 in the archived 3:2 counterbalanced order;
each arm gets a fresh server and the exact 12 serial requests. The full study
has a 7,200-second absolute deadline with 100-second final reserve, no retries,
replacement pairs, extra warmups, cache flushing or profiling timings.

There is **no NAS-backend-data reproduction promise**: only shared client mount
counters were collected. Startup/shutdown PID-I/O phase data were not collected
and cannot be recovered from these exited processes. A future richer collection
would be a different declared protocol, not a repair of this immutable study.
