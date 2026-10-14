# Single-GPU Mixed / vLLM M7 experiment

This is a paper-specific **thin orchestrator**, not a replacement benchmark or
inference implementation. It imports the maintained
`../../../scripts/bench_ferrule_vllm.py` HTTP/SSE client, readiness and root-shutdown
helpers, and `../../../scripts/test_vllm_qwen35.py::run_process` authenticated
subreaper supervisor. It does not copy their implementations, edit vendor code,
monkeypatch inference, install anything, restore `prepare_bench_stage`, or use the
old TP2/EP2 exact-comparison adapter.

All reports go in new `runs/<id>/` directories here. Final receipts use exclusive
creation. Existing run IDs are rejected. The maintained HTTP client checkpoints
its **own new case's** `http.json` during execution; it never resumes or overwrites
a previous experiment. Large binaries/build products remain under `target/`,
referenced by absolute path and SHA256. `/papers/` is ignored by the repository;
back up these local artifacts separately. No gitignore or files outside this
folder are edited by preparation. Preparation/build can create new immutable
staged artifact directories under `target/paper-m7-release-artifacts/`; they never
overwrite an existing stage. Optional compilation writes to the specified Cargo
target directory and Cargo's normal caches.

## Safety and current evidence

- Default `plan` and `prepare` are CPU-only. They read metadata/source/artifact
  hashes, not model tensor payloads. They do not run vLLM `--help`, import Torch,
  CUDA or NVML, query GPUs, launch servers, or compile.
- Use the **base Python with `-S -B`**. The installed vLLM Python's normal startup
  executes a `.pth` CUDA namespace hook even before application code; `-S` avoids
  site hooks for orchestration. GPU runtime processes use their real installed
  entrypoint only after explicit permission, without `-S`. CPU orchestration is
  restricted to the exact base interpreter, rejects GPU-bearing base `.pth` hooks,
  disables user-site and PYTHONPATH pollution, and pins base startup hooks. The
  maintained helper's ordinary Python startup therefore never uses the vLLM venv.
- `build` is separately opt-in, GPUs hidden, SM86 explicitly configured, offline
  Cargo, no model inference. No build has been run merely by writing this runner.
- `feasibility` and `run` require all three: `--sole-gpu-owner-granted`, a literal
  `--gpu-uuid`, and nonempty `--owner-note`. This is an operator attestation, not a
  mechanism for acquiring a GPU reservation. Never supply it without the owner
  grant. Only the selected physical UUID is exposed to each server (logical GPU0).
- A nonblocking `fcntl.flock` per UUID is acquired before the first idle query and
  held through child custody cleanup and final idle checks. It serializes these
  local runners only: **not a global reservation** against unrelated GPU users.
  A worker checks that its live coordinator still holds the bound lock inode.
- Sealed, read-only launch receipts bind a random launch token, the live
  coordinator identity, prepared/build/quality/feasibility hashes and executable
  pins. Direct receipt-free `--_worker` invocation is rejected. The worker verifies
  the maintained custodian's journal and kernel PID/birth identity, then rechecks
  contracts immediately before launch. The Ferrule binary executes through a
  hash-verified descriptor. The Python vLLM console script uses its unique read-only
  staged path, compared with the open hashed inode immediately before launch: Python
  multiprocessing spawn must reopen a stable main-script path, not `/proc/self/fd`. Read-only
  modes are not protection against a hostile same-UID/root actor.
- Identity mode is versioned, never auto-selected. The default `m7.json` retains
  **hostpid_verified**: GPU apps must map to authenticated server descendants in
  the known initial Linux PID namespace. Nested namespaces still block this mode.
- `m7-device-exclusive.json` is the separately authorized v2 protocol:
  **device_exclusive_local_custody**, requiring the matching explicit
  `--identity-mode device_exclusive_local_custody` and operator grant. It separates
  kernel-authenticated local custody from whole-device observation. Two finite
  all-eight-GPU idle samples are mandatory before and after each case (no apps,
  utilization0, device memory0). Only the selected UUID is exposed to inference.
  Unmapped NVML PIDs and PID-set changes are retained, not interpreted as verified
  descendants. Multiple vLLM processes are not automatically foreign workloads.
  When host mapping is provable the strict foreign-process check still applies.
  **Absence of interference is an exclusive-allocation assumption, not proven.**
  GPU memory is whole-device (conservatively includes any interference), never an
  owned-process peak; `owned_process_gpu_peak_bytes` remains null. PSS/RSS are the
  actual locally visible owned server tree. No PID offsets or foreign signals.
- V2 feasibility records both frozen engines even if the first is infeasible,
  but only after authenticated drain and independent all-device idle. Failure or
  unknown custody/idle still blocks the next engine. No retries or policy changes.
  Original blocked receipts are historical evidence and are never rewritten.
- Authenticated kernel-ECHILD descendant custody and independent GPU-idle evidence
  are both mandatory after each case. Exit code zero alone is insufficient.
  There is no foreign-PID kill, daemon reuse, auto-retry, extra GPU, TP change,
  prefetch substitution, CPU inference fallback, or precision substitution.
- Authoritative historical quality is
  `target/validation/bf16-attention-model-current-20261007/FINAL.json`:
  original full40 and all 27 fresh teacher-force plus free-greedy/reset/cold PASS.
  The subsequent composition refactor was reported numerically unchanged.
  **This is not a stale fresh-quality failure.** The new release executable still
  needs its own basic-quality receipt before the Hello32 pilot; feasibility alone
  is lifecycle evidence, not numerical acceptance.

## Fixed protocol and resource meaning

`m7.json` and `m7-device-exclusive.json` are frozen, checksum-bound protocols; arbitrary overrides
are rejected. A different context, GPU size, host budget, workload, backend or
policy needs a reviewed versioned config/runner change and fresh receipts.

| Field | Contract |
|---|---|
| Checkpoint | `/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8`, text-only full40 |
| Topology | One physical GPU, TP1/PP1/EP1/DP1 |
| Seed / request | 1; literal raw `Hello`; expected input usage 1; greedy temperature0/n1 |
| Context / scheduling | 128 context, 8 scheduler tokens, 2 resident sequences, client concurrency1 |
| Boundaries | BF16 activation/KV/convolution; F32 SSM; numeric FP8 weights |
| EOS / prefix | Natural EOS retained; no ignore-EOS trick; prefix cache off |
| Feasibility | One Hello request, max2 output, no warmup, each engine in a new process |
| Pilot (`run`) | Max32 output, one warmup then four fixed-work requests, each engine in a new process |
| Observer policy | Baseline `FERRULE_LOG=info`; optional separately labeled diagnostic `info,ferrule_performance=debug` with JSON logs |
| GPU target | 7 GiB = 7,516,192,768 bytes, **soft/observed**, not a hard partition |
| Host target | 64 GiB = 68,719,476,736 bytes, sampled process-tree PSS; RSS also retained |
| Ferrule expert cache | 320 MiB, eight entries; includes 8 MiB expert scratch |
| Ferrule workspace | 2 GiB operator grant, excluding expert scratch |
| Ferrule attention KV | 16 MiB; recurrent state is separately accounted |
| vLLM policy | UVA offload30 GiB of `w13_weight`/`w2_weight`, scales not explicitly offloaded; eager |

On the historically recorded 24-GiB device, vLLM utilization `7/24` is a planning
fraction, not allocator enforcement. The runner requires telemetry total to equal
24 GiB rather than silently using that fraction on a different GPU. Device-wide
memory includes the idle baseline. Observations exceeding 7 GiB or 64 GiB PSS
terminate the owned server root and invalidate the case. Copy/repack/JIT
transients may fail; the result remains a failure, never a silent budget increase.

There is **no RLIMIT_AS** (CUDA virtual reservations are not physical residency),
no cgroup memory limit creation, no hard GPU partition, and no claim of exact
instantaneous peaks. Each telemetry sample uses finite `nvidia-smi` invocations,
not an unbounded `watch` process, at nominal 2-second intervals plus query cost.
Missing/error telemetry is not zero. Device samples, process RSS/PSS and
`VmLck`/`VmPin` observations are retained. The latter do not measure all CUDA
pinned memory; `pinned_bytes` stays null. PSS excludes unmapped OS file cache,
so `observed_within_targets` is **not proof of a total physical-host-memory cap**.
Cold means a fresh process, not cold disk/page cache; no global drop-cache.

Same numeric boundaries are not identical arithmetic: Marlin packing, scale
conversion, reductions and GDN kernels differ. Ferrule's cuBLAS `:0:0` workspace
policy differs from vLLM's clean-environment default, deliberately reported as a
system policy difference, not a dtype-boundary mismatch. This experiment always
reports `exact_speedup: null`, `fairmatch: false`. Complete descriptive observations
can coexist with blocked effective-runtime/precision/resource fairness.

`max_tokens=32` does not promise 32 actual tokens. Natural short EOS remains a valid
request; actual counts and length-match flags come from the maintained client.
Do not compute length-matched speedups from unequal outputs. Input128/output32 is
**deferred**: it requires ctx160 and new admission/basic quality, not truncation.

## Commands from repository root

### 1. CPU plan (safe now)

```sh
env CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=none \
  /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
  papers/arxiv/experiments/runner.py plan --id m7-plan-001
```

Omit mode and `--id` to print the plan without creating artifacts. Plans include
actual model metadata/stat identity, authoritative quality metadata, CPU monotonic
clock information, memlock limits, source-only CLI audit and exact build argv/env.

### 2. Optional bounded release build (CPU only; run only when assigned)

```sh
env CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=none \
  /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
  papers/arxiv/experiments/runner.py build --execute-build --id m7-build-001
```

This submits a literal external Cargo command through `run_process`, not a shell:

```text
/root/.rustup/toolchains/nightly-2026-08-01-x86_64-unknown-linux-gnu/bin/cargo build --offline --locked --release -p ferrule-cli
  --bin ferrule --no-default-features --features cuda --message-format=json
  --target-dir /home/admin/luhuanbing.lhb/Ferrule/target/paper-m7-release
```

The frozen CUDA12.4 environment comes from the existing build-environment JSON.
Release overrides explicitly require opt3, fat LTO, one codegen unit, no debug
assertions, `CUBLAS_WORKSPACE_CONFIG=:0:0`, and SM86. No historical opt1 debug
executable is accepted. Build worker timeout is 1500 s plus bounded supervisor cleanup;
use a terminal ceiling of 1,800,000 ms including CPU inventory hashing.
a timeout is reported, not retried. Cargo JSON must identify exactly one non-test
CUDA executable. A verified read-only copy in a unique target artifact stage, not
the mutable Cargo output path, is the executable bound to later receipts. Sources before/after must match. Hashes bind tools, CUDA libraries,
source files, workflow and binary. Mandatory inventories are recomputed for coverage,
not merely checked against whatever a receipt supplied: host/CUDA compiler programs,
full selected Rust toolchain, CUTLASS tree, CUDA headers/libdevice/shared libraries,
and CPU Python startup hooks. Preparation additionally inventories installed Python
sources, Torch/vLLM and other shared libraries, `.pth`, package METADATA/RECORD and
JIT compiler executables. Missing inventory groups or added/changed files block
validation. These are complete **declared input inventories**, not a claim of full
checkpoint identity or complete dynamically generated JIT/library closure.
An external build owner may provide the same
`ferrule.paper.m7.build.v1` receipt contract, with the exact command/environment,
Cargo artifact and authenticated supervisor evidence; a bare binary is insufficient.
No release build, linker compatibility, or device fit is claimed until actually run.

### 3. CPU prepare after build

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
  papers/arxiv/experiments/runner.py prepare --id m7-prepared-001 \
  --build-receipt papers/arxiv/experiments/runs/m7-build-001/build.json
```

Preparation pins current sources, executable, helpers, vLLM installed source/shared
objects, metadata, and JIT environment without importing the runtime. Any changed
pin blocks launch; reprepare/rebuild instead of patching historical receipts.
Paper-only orchestration changes require fresh preparation, not recompilation of
an unchanged Rust executable. Historical build-workflow hashes remain in the
build receipt; the new manifest pins current paper scripts, while current Rust
source, build environment, compiler/library inventories and staged executable
must still match. Large installed-runtime inventories are checked at launch, not
per HTTP request.
The checkpoint identity is explicitly **partial**: config/index/tokenizer hashes
plus shard file stats. `full_checkpoint_sha256` is null. No 37-GB payload hash is
performed, and metadata identity must not be relabeled strong checkpoint identity.
A later separately assigned, bounded CPU payload-hash job can supply stronger
identity without re-reading the checkpoint for every engine.

### 4. Feasibility (ONLY after actual sole-owner permission)

The UUID below is historical GPU0 identity, not a claim of present availability.
The owner must confirm it is the granted physical device and replace the owner
note with the actual reservation reference. Supplying these flags is authorization.

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
  papers/arxiv/experiments/runner.py feasibility --id m7-feasibility-001 \
  --prepared papers/arxiv/experiments/runs/m7-prepared-001/prepared.json \
  --sole-gpu-owner-granted \
  --gpu-uuid GPU-b980f22b-58a3-fc88-21c1-ae86053d8535 \
  --owner-note 'OWNER MUST SUPPLY THE ACTUAL GRANT REFERENCE'
```

Before each engine: revalidate pins, collect two independent idle samples (no
compute apps, utilization0, <=512 MiB background device allocation), verify total
memory, and check loopback port availability. Observe both engines sequentially;
first failed case stops the pair. Every server is a foreground owned child.

### 5. Release basic-quality receipt, then Hello32 pilot

An authorized quality owner must provide `ferrule.paper.m7.release-quality.v1`
(`schema.json`), tying these exact fields from `prepared.json`:

- `binary_sha256`, `model_identity_sha256`, `protocol_sha256`;
- `status: "pass"`, `independent_reference: true`;
- `checks`: `raw_hello_greedy`, `reset_repeat`, `cold_repeat`, all true;
- `artifact_pins`: at least the independent reference and actual new release
  result artifacts, absolute paths to files and their SHA256.

The owner must actually compare the newly built public CLI's greedy raw completion
against its independent expected result, repeat after request reset and a cold
process restart, and preserve input/output/count evidence. This runner **checks
binding and completeness of the owner attestation**, not the semantics of arbitrary
external quality artifacts. It does not fabricate a pass from historical quality,
HTTP200, agreement between the two engines, or the two-token feasibility request.
Place the owner receipt and paper-specific results in a new assigned run directory.
It is distinct from full logits/routes/state numerical acceptance.

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
  papers/arxiv/experiments/runner.py run --id m7-hello32-001 \
  --prepared papers/arxiv/experiments/runs/m7-prepared-001/prepared.json \
  --feasibility papers/arxiv/experiments/runs/m7-feasibility-001/execution.json \
  --release-quality papers/arxiv/experiments/runs/m7-release-quality-001/quality.json \
  --sole-gpu-owner-granted \
  --gpu-uuid GPU-b980f22b-58a3-fc88-21c1-ae86053d8535 \
  --owner-note 'OWNER MUST SUPPLY THE ACTUAL GRANT REFERENCE'
```

This is a four-request c1 pilot, not a throughput scaling study or statistically
powered speedup result. Each engine uses a fresh process, fixed Ferrule-then-vLLM
order. Do not treat same-server requests as independent process replicas. A later
study should preregister counterbalanced fresh-process repetitions separately.

## Bounds and receipts

| Phase | Maximum seconds |
|---|---:|
| Startup/JIT | 900 |
| Each HTTP request | 120 |
| Measured fixed-work window | 360 |
| Graceful root cleanup | 120 |
| Authenticated supervisor case timeout | 1600 |
| Each independent before/after idle gate | 20 |
| Additional supervisor handshake/cleanup reserve | 60 |
| Declared execution envelope per engine | <=1700 |

CPU source/library pin validation occurs before the timed server case. These are
process/deadline bounds, not a guarantee of interrupting a stuck driver/kernel.
Unkillable/unknown custody or release is a failure that forbids the next engine.
The HTTP request deadline does not establish server drain.

Each case retains literal isolated argv/environment, metadata, startup log,
`samples.jsonl`, HTTP report, root shutdown result, effective-runtime log evidence,
authenticated supervisor result, before/after GPU UUID/driver/PCI/memory evidence,
and an assessment. The top-level `execution.json` hashes artifacts and refuses
exact claims. Server log observation is capped at 64 MiB (sampled, not an atomic
filesystem quota); runtime evidence scans a bounded prefix and labels truncation.
The maintained supervisor preserves bounded stdout/stderr tails.

The client uses host `CLOCK_MONOTONIC` where provided; nominal resolution is not
accuracy. HTTP TTFT is send to first nonempty SSE content; SSE content-event ITL is
not token-level device ITL. No GPU event timings are invented. Missing backend,
dtype, offload-mode or pinning evidence keeps runtime fairness blocked. Known
fallback/contradiction messages invalidate the case; there is no patched vendor
instrumentation. UVA selection alone cannot prove zero-copy was actually used.

## Source/parameter evidence

vLLM paths below are beneath
`target/vendor/vllm028-py312/lib/python3.12/site-packages/vllm/`.

| Configuration | Actual implementation / evidence |
|---|---|
| All Ferrule serve flags, units and restrictions | `crates/ferrule-cli/src/args.rs:330–568`; `crates/ferrule-cli/tests/support/mixed_serve_smoke.json` |
| Named Mixed boundaries/lazy/CUBLAS startup | `crates/ferrule-composition/src/model_factory/mixed_serving.rs`; CLI `serve.rs` |
| Release opt3/fat-LTO | root `Cargo.toml` `[profile.release]`; explicit build env; Cargo artifact receipt |
| vLLM host/port/name + parallel/model flags | `entrypoints/openai/cli_args.py`, `engine/arg_utils.py`; `config/model.py` rejects spelling `bf16` |
| BF16 attention KV, Mamba conv/SSM | `config/cache.py`; `models/qwen3_5.py::get_mamba_state_dtype_from_config`; `layers/mamba/mamba_utils.py` |
| UVA30 selective routed weights | `config/offload.py`; `model_executor/offloader/uva.py`; segment names `w13_weight`/`w2_weight` |
| GPU rather than CPU inference | UVA CUDA views or functional-call H2D fallback; `--offload-backend` is a storage policy |
| Explicit MoE Marlin selection | `layers/fused_moe/oracle/fp8.py`: unsupported explicit choice raises; Marlin selects W8A16 |
| FP8 Marlin capability | `layers/fused_moe/experts/marlin_moe.py`: CUDA >=7.5 and static128-block FP8 accepted |
| Linear fallback caveat | `model_executor/kernels/linear/__init__.py`: backend without a kernel category warns/falls back |
| Repack/offload transient caveat | `model_executor/model_loader/utils.py::device_loading_context`; `quantization/utils/marlin_utils_fp8.py` |
| Batch8 with ctx128 | `config/scheduler.py::verify_max_model_len`: enable chunked prefill when batch<context |
| GPU utilization is planning, not reservation | `v1/worker/utils.py::request_memory`; `v1/worker/gpu_worker.py` |
| No fixed vLLM KV bytes | `--kv-cache-memory-bytes` bypasses utilization sizing; Ferrule16 MiB is not a hybrid-cache equivalent |
| Prefix/EOS/generation/eager | `config/cache.py`, `config/model.py`, `engine/arg_utils.py`; client `make_payload` and actual usage |
| CUDA12.9 JIT environment | `target/validation/vllm028-py312/jit-cuda12/fix-env.json` |
| Historical JIT/driver failure and fix | neighboring `result.json`, `cu129-complete-result.json`: old CUDA13.2 cached sampling vs driver API12090; fixed CUDA12.9 PASS |

The vLLM server gets an `env -i` environment plus that **exact fix JSON**; no inherited
CUDA12.4 `LD_LIBRARY_PATH`, `LD_PRELOAD`, Marlin A8 override, offload-disable flags,
or Ferrule cuBLAS setting. The future GPU query records current driver identity;
historical compatibility is not current device validation. Requested settings are
never promoted to effective settings merely because they appeared in argv.

## CPU validation

```sh
env CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=none PYTHONDONTWRITEBYTECODE=1 \
  /home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover \
  -s papers/arxiv/experiments -p 'test_*.py' -v
```

Tests include actual installed CLI AST/declarations, schema rejection, changed
parameters/pins, release-vs-debug, authorization-before-query, no heavy runtime
imports, EOS/count failure semantics, missing telemetry, unknown custody,
fail-stop orchestration mocks, and a real **CPU-only** supervised child. No GPU
server is used. JSON Schema validation uses an already installed CPU validator
when present, adding its package path without running site hooks; no installation.
`contract.py` enforces semantic hashes/custody/budget conditions beyond JSON Schema.
Prepared receipts require the actual fields consumed at launch, 64-hex digests and
nonempty pin maps. Complete executions require exactly one Ferrule and one vLLM
case. `run` always binds non-null feasibility and release-quality receipts.

### Review fixes and diagnostic separation

The static CLI audit is a scoped validator for the 25 concrete vLLM options in this
protocol, using installed AST type/enum declarations. It rejects `bf16`, malformed
or nonfinite numeric values, malformed offload lists, unknown switches and boolean
negation of non-bools (such as `--no-cpu-offload-gb`). It does not pretend to be a
general vLLM parser. The JIT environment accepts **exactly** the reviewed keys in
`fix-env.json`; visibility/preload/offload/precision overrides are rejected, and
selected UUID/offline assertions are checked after merging.

`--diagnostic-performance` is an explicit separate observer profile. It selects
`FERRULE_LOG=info,ferrule_performance=debug` with JSON logging; ordinary baseline runs
remain `info`. Diagnostic receipts cannot qualify a baseline feasibility gate.
The current runtime parser still only extracts profile/backend evidence and emits
`performance_fields_observed: false`; enabling debug does not invent measurements.
An in-flight telemetry failure is always persisted even if `finish()` concurrently
requests stop. Missing/unknown custody, memory, namespace or query evidence is not
converted to zero or success.

## Explicit device-exclusive v2 invocation

Use `--config papers/arxiv/experiments/m7-device-exclusive.json` for both fresh
`prepare` and `feasibility`, plus
`--identity-mode device_exclusive_local_custody` on `feasibility`/`run`.
The strict default cannot silently consume the exclusive prepared manifest.
The v2 config changes identity semantics only; all model, precision, backend,
offload, workload, resource and timeout settings are identical to v1.
Execution receipts use `ferrule.paper.m7.execution.v2`, bind the prepared protocol,
and label performance as `exclusive_operator_allocation_assumption`.
Stage startup time excludes separately reported CPU contract validation. Library
inventories are verified immediately before each server launch, never per request.

The first v2 pair (`m7-feasibility-exclusive-001`) records a genuine Ferrule Hello2
result and a vLLM launch-wrapper failure: multiprocessing could not reopen the
fd-based main script. This is **not a vLLM memory-fit result**. The spawn-compatible
paper-only launch fix preserves model argv/environment and staged bytes; its real
CPU multiprocessing regression precedes a fresh prepared manifest and new pair ID.

The second pair (`m7-feasibility-exclusive-002`) completed Ferrule Hello2 and
observed vLLM exceed M7 during startup. Its append-only assessment documents an
over-broad historical fallback classifier (the Triton GDN default is not a Marlin
or UVA switch) and a readiness-poll delay after monitor termination. The current
runner preserves GDN fallback observations without asserting an offload change,
and aborts readiness when the resource monitor fails so cleanup begins promptly.
Neither correction makes the recorded vLLM case eligible, and no GPU rerun or
resource-policy substitution follows these evidence/cleanup fixes.
