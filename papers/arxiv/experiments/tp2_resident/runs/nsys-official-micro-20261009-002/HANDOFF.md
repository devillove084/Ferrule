# Official Nsight installation + dual-GPU micro: PASS

No full-model run was started. No production, torch, vLLM, CUDA installation, or old 2026.1.3 raw was modified.

## Installed tool

- NVIDIA Nsight Systems CLI **2025.3.1.90-253135822126v0**; matching QdstrmImporter from the same official package.
- Project-relative CLI: `target/vendor/nsight-systems-cli-2025.3.1/opt/nvidia/nsight-systems-cli/2025.3.1/target-linux-x64/nsys`
- Importer: same install root, `host-linux-x64/QdstrmImporter`.
- Both sample processes actually loaded the bundled `target-linux-x64/libcupti.so.12.9` (SQLite diagnostic evidence).
- Official source: https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/NsightSystems-linux-cli-public-2025.3.1.90-3582212.deb
- Package: 175210028 bytes; SHA256 `d2484ad0faf6831b11fa0bf73c54232d9ea8beafb50414019e6ba299c4ed5718`, matched NVIDIA repository Packages index.
- Isolated extraction, not a system install. Declared dependencies: libc6, libglib2.0-0. No missing ldd dependencies. Details: `installation.json`, `dependency-sha256.json`, `tool-inventory-sha256.json`, `*-ldd.txt`.
- NVIDIA HTTPS + repository hash verified; package signature was not independently verified.
- `/opt/conda310/bin/conda` 23.1.0 / Python 3.10.9. Official NVIDIA conda exact-package dry-run returned PackagesNotFoundError; no conda transaction performed. Evidence in sibling `nsys-official-micro-20261009-001/research/conda-dry-run.json`.
- Host tested: 8x RTX3090 SM86, driver 575.57.08 / driver API 12.9. Existing CUDA12.4.99 official demo_suite/vectorAdd used; release notes for 2025.3 explicitly list CUDA12.9 support.

## Successful command

From the project root (this run directory is now immutable; do not rerun into it):

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/tp2_resident/nsys_micro.py run papers/arxiv/experiments/tp2_resident/runs/nsys-official-micro-20261009-002 --sole-gpu-owner
```

Exact expanded collector command/environment: `runs/P1/session/collector-command.json`.

Capture options: `profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --gpu-metrics-devices=none --gpuctxsw=false --duration=150 --kill=none --wait=all --stop-on-exit=true --export=sqlite --force-overwrite=false`.

2025.3 uses application-scoped `cuda`; do not copy 2026-only `cuda-sw`, `--cuda-trace-scope`, `--cuda-trace-all-apis`, or `--gpu-video-devices` options into this command. Any full-model adapter needs the appropriate version-specific options and a fresh grant/run, not a blind path substitution.

## Acceptance

`acceptance.json` is PASS, not merely exit-zero/API-only:

- 2 vectorAdd kernel activity rows, one on each physical UUID:
  - GPU0: `GPU-b980f22b-58a3-fc88-21c1-ae86053d8535`
  - GPU1: `GPU-63f1bd01-ddf9-a742-ee48-9389892493d9`
- 6 memcpy rows: each GPU has two 200000-byte H2D copies and one 200000-byte D2H copy.
- 26 runtime/API rows; every kernel and memcpy has a successful correlated API.
- Both official vectorAdd executions report `Test PASSED`.
- CLI/importer successful; no fatal SQLite diagnostic, NumTpcs, or StringStorage error.
- The two sample processes each see a singleton CUDA device, so raw kernel `deviceId` is 0 in both. Correct mapping is `globalPid -> PROCESSES.pid`, then `(pid,deviceId) -> TARGET_INFO_CUDA_DEVICE.(pid,cudaId) -> gpuId -> TARGET_INFO_GPU.uuid`. Physical IDs are **0 and 1**. Direct joining kernel.deviceId to physical GPU.id is invalid here.
- Nonfatal warnings are retained: CPU scheduling disabled; no NVTX in sample; the CPU-only Python gate has no CUDA events/CUDA-start warning. Neither actual vectorAdd process has that CUDA-start warning.

Artifacts: `runs/P1/session/micro.nsys-rep`, `runs/P1/session/micro.sqlite`, `runs/P1/session/server.log`, `acceptance.json`, `artifact-sha256.json`.

Offline acceptance command already run successfully:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/tp2_resident/nsys_micro_accept.py papers/arxiv/experiments/tp2_resident/runs/nsys-official-micro-20261009-002
```

The validator is exclusive-create for its receipt; use its `accept(connection)` for read-only rechecks rather than overwriting `acceptance.json`.

## Custody / GPU release

- Existing authenticated Linux subreaper supervisor + sorted eight UUID locks; workload restricted to GPU0/1.
- B0 all eight zero -> collector-only B1, 3 equal samples over 4.47 s -> selected work -> two exact B1 return samples -> collector authenticated drain -> all-eight original B0 -> release.
- B1: GPU0/1 4 MiB each, other GPUs 1 MiB each; unchanged 16 MiB instrumentation ceiling and 1 MiB nonselected delta tolerance.
- `gpu-release.json`: authenticated_drained=true, supervisor_success=true, original_b0_verified=true, b1_returned=true, local_contexts_closed=true.
- `final-idle.json`: all eight 0 MiB / 0% and no compute processes after release.
- GPU ownership is released. No foreign PID signals were issued; only existing authenticated custodian cleanup was used.

## Tests and prior attempt

- 12 dedicated CPU tests passed (`accept-cpu-tests.log`), including logical-to-physical mapping, API-only rejection, missing/ambiguous mapping behavior, failed API, missing D2H, fatal diagnostic, preserved injection, unchanged scope, and monitor-event regression.
- The separate existing instrumentation suite had 13 scope tests pass, but its live test module could not import because existing `paired.py` imports missing `observed`; not modified here. This micro adapter does not import paired.py. Original failure is preserved in sibling run001 `cpu-tests.log`.
- Run001 is not accepted: the new adapter initially called nonexistent monitor.is_set(), causing worker failure despite valid profile data. Fixed to monitor.failed.is_set(), regression tested, and run002 was fresh. Old receipts/raw were not changed.

This establishes official Nsight micro compatibility, not full-model coverage or performance. Stop here and hand off for the separately authorized full-model profile.
