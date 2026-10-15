# TP1 official read/compute overlap profile v1

> **Current plan: generation -04.** See [GENERATION_04.md](GENERATION_04.md)
> for the owner-safe traced wrapper, process-tree context switches, independently
> bound model/wrapper roles, scoped versus full-trace acceptance, and fresh
> parent GO after pressure cleanup. The original protocol below is historical;
> its `--cpuctxsw=none`, older lifetime text and preparation commands do not
> authorize -04. Generations -01/-02/-03 remain rejected/blocked, unchanged.

**CPU preparation only until the parent explicitly says GO.** This is a separate
profile, never an extension of ordinary paired timing. No profiler capture,
model launch, GPU query, lock acquisition, cache flush, reset, or foreign-process
signal is part of `plan`, `prepare`, `check`, tests, or offline acceptance.

## Frozen scope and reuse

- Exactly one pair, order **blocking → lookahead**, registered before capture.
  One existing `canary-qa` request (`The capital of Japan is`, 8 tokens) after
  Ready in each arm. No Hello, warmup, conditioning request, replacement, retry,
  or timing sample. `/v1/models` readiness polling is not a generation request.
- Same immutable CLI: commit `86c58a017d7c4b53490f536a3b1a133f2f7848cc`, SHA256
  `7cc8e0ff6e84f27090494b8e9e27dc3a388d1d2f3fc34ae39dae416e1e607a01`.
  The actual build has `cuda,ferrule-common/nvtx-profile`; runtime NVTX remains
  OFF in ordinary timing and is `1` only in these profile environments.
- `protocol.py` reruns the maintained final-layout **raw build and nine-gate
  reviewer**, then seals a new, explicit TP1 `candidate-binding.json`. It does
  not rewrite or inherit acceptance from the historical paired binding. That
  binding's README pin was stale during preparation; no old evidence was fixed
  or suppressed. The model arguments and limits are the existing `cases-v3.json`
  TP1 case, with no generic factory or cross-block TP2 substitution.
- `tp1_collector.py` uses private instances of maintained
  `tp2_resident/instrumentation_live.py`, `instrumentation_scope.py`, and
  `instrumentation_capture.py`. Only the explicit one-UUID grant and finite
  capture-wrapper budget are specialized. The maintained authenticated
  `launch_guard`, `_owned_subreaper.py`, sorted eight UUID locks, stable B0/B1
  policy, observer, and `_profile_capture.run_target` own the actual lifecycle.
  The historical two-GPU modules are not mutated.
- Model visibility is **only** `GPU-b980f22b-58a3-fc88-21c1-ae86053d8535` (TP1,
  PP1, EP1). The eight locks cover the separately granted profiler enumeration
  scope; they do not widen model visibility. Physical queries occur only after
  GO and successful lock acquisition. An exclusive `capture-claimed.json`
  prevents another pair under this version even with another output directory.

## Tool and finite lifetime

Actual installed target (version/help checked, not a test capture):

```text
target/vendor/nsight-systems-cli-2025.3.1/opt/nvidia/nsight-systems-cli/2025.3.1/target-linux-x64/nsys
NVIDIA Nsight Systems version 2025.3.1.90-253135822126v0
```

Capture flags:

```text
profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
--gpu-metrics-devices=none --gpuctxsw=false --osrt-file-access=true
--osrt-threshold=1000 --duration=150 --kill=none --wait=all
--stop-on-exit=true --export=none --force-overwrite=false
```

Export is separately bounded and explicit:
`export --type=sqlite --lazy=false --force-overwrite=false`.
No unsupported `cuda-trace-scope`, CPU sampling, or graph-node tracing is used.
The installed help says file-access tracing ignores `osrt-threshold` **for file
APIs**; the 1000ns threshold still applies to other OSRT calls and is reported.

Startup has a fixed 120s deadline from collector launch (allowing slow profiler
startup, not the ordinary 20s assumption). QA has at most 20s and must finish
before the finite capture cutoff. The direct child has a 150s lifetime and 30s
shutdown reserve. The worker uses the maintained stop-marker → direct child
termination/reaping → B1 return → natural collector exit handshake, with a 100s
cleanup/export wait and a 300s authenticated supervisor bound. SQLite export has
60s inside that supervisor bound. A timeout is a failed attempt, not a reason to
reset clocks or collect again. Startup/collector/export cost is preserved; none
is discarded or substituted for ordinary performance.

## CPU commands

From the repository root:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/profile_overlap_v1 -p 'test_*.py' -v
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/profile_overlap_v1/run.py plan
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/profile_overlap_v1/run.py prepare --directory papers/arxiv/experiments/storage_demand/profile_overlap_v1/artifacts-cpu-ready-20261015-01
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/profile_overlap_v1/run.py check --directory papers/arxiv/experiments/storage_demand/profile_overlap_v1/artifacts-cpu-ready-20261015-01
```

`prepare` requires a fresh directory and performs only CPU file hashing, ELF
inspection, maintained raw qualification review, and actual Nsight help/version
commands. After any source changes, prepare a **new** directory; do not rewrite
an old seal. The private output is mode 0700 and artifacts are private by umask.

## Parent GO / ready hook

The `run` command remains gated. **Do not run it while ordinary paired timing
owns the GPUs.** The parent must first close ordinary timing and review any
startup/configuration failure. A failed ordinary study does not permanently
ban this separate profile; its failure closeout must remain pinned, the cause
must be handled by the parent, and no ordinary acceptance is inherited.

The parent supplies an immutable/read-only JSON receipt with these exact fields:

- `schema`: `ferrule.storage-demand.profile-overlap.v1.parent-go`
- `decision`: `GO`
- `directory`: absolute path of the prepared profile directory
- `prepared_sha256`: SHA256 of that directory's `prepared.json`
- `pair_ceiling`: `1`; `order`: `["blocking", "lookahead"]`
- `ordinary_timing_closed`, `sole_gpu_owner`, `instrumentation_all_eight`: `true`
- `ordinary_closeout`: `{ "path": absolute closeout path, "sha256": its hash }`
- `boot_id`: actual `/proc/sys/kernel/random/boot_id`
- `expires_unix_ns`: parent-chosen bounded expiry

Only after that separate authorization, the parent invokes `run.py run` with
`--directory`, `--parent-go`, `--parent-go-sha256`, and
`--sole-gpu-owner-granted`. No GO receipt is generated by preparation. Internal
`worker`/`target` entrypoints authenticate the maintained launch journal,
coordinator/helper/worker kernel identities, locks, nonce, and collector gates;
invoking their names alone is not authorization.

After both captures and authenticated drain, the CPU-only acceptance command is
`run.py accept --directory` with that same directory. Acceptance is exclusive
and fails closed; no GPU launch is possible through it.

## Evidence and acceptance

`run.py` binds the actual **inner model child** PID/birth/parent, not the outer
nsys PID. At Ready and after QA it calls the maintained `7d3bee2`
`runnerhelpers/library_maps_v1.py` private snapshot/validator. All original CLI
runtime requirements remain required; pinned Nsight injection/CUPTI and their
ELF dependencies are added explicitly. A CPU role is never substituted. Raw
maps survive later validation failures. Pathname/inode evidence is not a live
mapped-memory hash; overlay ambiguity remains explicit.

The analyzer reads only SQLite `mode=ro`:

- Supports the observed official 2025.3 schema **3.20.2** only; lazy, filtered,
  shifted, missing, or unknown exports reject.
- Resolves `NVTX_EVENTS` domain-create and start/end event kinds using enum
  tables, direct `text` and `StringIds`. Only the `ferrule_profile` domain and
  exact numeric labels qualify. `on_new_span` records numeric fields at span
  creation; on-enter emits `read_subset operation=… source_generation=…
  requested_bytes=…`, `layer.expert image_generation=… layer=… expert=… rows=…`,
  and the separate `layer.expert.upload` label. These existing ranges need no
  production changes.
- Retains the VM/process bits of serialized global IDs; PID decoding is
  `(globalId >> 24) & 0xffffff`. Correlations use **full process key + correlation
  ID**, then API thread/time containment. Physical UUIDs come from
  `PROCESSES` → `(pid,cudaId)` → `gpuId` → `TARGET_INFO_GPU`, never from assuming
  logical ordinal zero is physical GPU zero.
- Requires bidirectional one-to-one math launch/kernel correlation across the
  full export, success returns, positive expert/read/upload ranges in both
  modes, and an actual kernel for every expert submission range. Copy APIs,
  upload ranges, waits and CPU envelopes cannot become GPU compute. Graph
  launches, nodes, ambiguous one-to-many mappings and orphans reject even when
  table row coverage looks complete.
- Computes the union of actual attributed kernel intervals and separately
  intersects it with (a) worker read envelopes and (b) actual worker OSRT
  `pread`/`pread64` intervals. Nested aliases/duplicate reads or PIDs reject;
  overlapping kernels count once. Worker pread bytes must reconcile per numeric
  read range, and read counts/bytes reconcile to current decoder deltas and
  final owner ledger, including the last source/operation receipt.
- Checks exact generated token IDs, HTTP text/usage, actual request/session ID,
  admission/terminal clocks, raw map snapshots, owner bookkeeping, supervisor
  drain, B1 return and original idle before accepting the pair. Raw report,
  SQLite, logs, command receipts, API counts, thresholds and warnings remain
  private at explicit paths in the analysis. No large kernel-row artifact is
  emitted.
- Preserves every diagnostic. Errors, lost/dropped events, partial collection,
  unknown warnings, and model CUDA-start/NVTX warnings reject. Only the exact
  known no-CPU-context-switch warning (not used for overlap) and the known
  no-CUDA/no-NVTX diagnostics on authenticated non-CUDA wrapper PIDs receive
  explicit benign classifications. CLI/model WARN/ERROR/FATAL also reject.

Synthetic `fixture.sql` exercises schema/interval logic, not a pretend capture.
The optional local historical tests read the accepted small two-GPU micro and
only NVTX schema/enum metadata from a historical model export. The micro lacks
Ferrule NVTX events and **cannot** certify this candidate's overlap.

Even accepted overlap describes one profiled QA, not a performance comparison,
physical NAS service time, or a full-memory bound. Page-cache behavior remains
uncontrolled. API/activity row closure alone cannot rule out simultaneous loss;
all independent evidence checks above remain mandatory.

`/papers/` is ignored by the repository's existing `.gitignore`. New source here
therefore needs explicit force-add if the parent later chooses to version it;
this preparation does not stage, commit, or alter the ignore rules. Do not add
private capture artifacts or the global `capture-claimed.json` attempt marker.
