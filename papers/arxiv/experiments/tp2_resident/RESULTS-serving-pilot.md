# TP2 release-serving smoke and fresh F/V QA8 pilot

**Completed, descriptive only.** Same-release serving smoke passed, followed by the planned fresh-server order `F1,V1 / V2,F2`. No retry, model change, HF regeneration, production edit, package install or transport fallback was used. `fairmatch=false`, `exact_speedup=null`; E2 remains incomplete.

## Admission and smoke

- Ferrule release SHA256: `e017a3a8c8ec4019a7f291a818643d9693e32b2dfee0a9cc503fcf84926c1b0b`.
- Model/runtime **test-binary** quality: full40 Hello/Capital each 2,178 exact fields; six lifecycle recipes total 33 calls / 35,937 exact metrics. Fresh `greedy4-02` closes provenance; old failed `greedy4-01` remains failed.
- Catalog build: pinned Cargo release artifact, 568 live production/build inputs, 12 dynamic-library pins, unchanged numerical/model/runtime inputs. Full 2,940-entry before/after build inventories are retained as immutable artifacts; generated nested targets and test-only files are not live execution dependencies. Build custody is **not authenticated**; launch custody is.
- Bootstrap did not require its own missing HTTP receipt. Both fresh smoke servers then passed Hello2, three frozen canaries, repeat canaries, intentional cancellation, post-cancel repeat canaries, host admission-accounting cleanup, exit 0, authenticated descendant drain and idle/release.
- S1/S2 startup: **369.969 / 370.751 s**. Each sampled GPU peak: **18.042 GiB/card**; PSS **49.238 / 49.237 GiB**. Separate identities prove a fresh CPU/server owner, not an unexposed in-process rebuild API.
- Cancellation matched request ID 8 in each server: `cancelled`, `cancellation_submitted=true`, `cancellation_failed=false`, `requires_restore_or_shutdown=false`. Partial client output usage remains null; worker generated 2 vs emitted 1 is retained, not forced into a normal completion.
- Smoke request DEBUG logs were diagnostic only; numerical/state trace reserves were zero. All smoke timing is excluded from the performance table.
- `smoke-001/002` were CPU preparations only, with append-only `not-run.json`; zero server starts. Actual smoke is `tp2-serving-smoke-20261009-003`.

## Fresh descriptive measurements

Each server: Hello2 → three original max8 canaries → QA8, concurrency 1, one excluded first QA-phase request + two warmups + five measured requests. All 48 normal paired requests passed; each measured block produced **40 actual output tokens (5 × 8)** from five-token Japan prompts. Natural EOS/greedy seed1 retained; measured finish reasons were `length`. The excluded QA request is not the first request after server readiness.

| Slot | Startup s | QA client wall s | Actual output tok/s | Mean TTFT ms | Mean E2E ms | GPU0 / GPU1 peak GiB | Tree PSS peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| F1 | 385.979 | 7.244697 | 5.5213 | 516.128 | 1441.267 | 18.0420 / 18.0420 | 49.2373 |
| V1 | 63.794 | 3.847396 | 10.3966 | 134.081 | 709.799 | 19.2842 / 19.2842 | 7.4963 |
| V2 | 60.095 | 3.787132 | 10.5621 | 130.161 | 695.507 | 19.2842 / 19.2842 | 7.0099 |
| F2 | 367.234 | 7.218413 | 5.5414 | 511.283 | 1437.886 | 18.0420 / 18.0420 | 49.2374 |

These observations do **not** show throughput parity: Ferrule is slower in this short workload. No speedup ratio, significance, optimality, long-output extrapolation or confidence interval is claimed. Two order-balanced blocks are not a broad statistical result; five same-server requests are not five independent replicas. Raw per-request first/warmup/measured text, usage, TTFT/E2E, client overhead and failure denominators are retained in each `http.json` and the summary.

## Matched scope and explicit differences

- Same physical GPU0/1 UUIDs: `GPU-b980f22b-58a3-fc88-21c1-ae86053d8535`, `GPU-63f1bd01-ddf9-a742-ee48-9389892493d9`; PP1/TP2/DP1, Ferrule EP1 and vLLM EP disabled. This is **not TP2EP2**.
- Same FP8 source checkpoint, BF16 activation/KV/conv boundary and F32 SSM, context128/batch8/active2/prefill8, prefix off and no CPU weight offload. Weight payload identity remains metadata/shard-stat based, not a full payload hash.
- Ferrule: thread CPU/host collectives and serial decode; actual scheduler lowers requested decode batch 2 to **1**, disables mixed batches and cohort deferral. CPU collectives are not CPU parameter offload. The HF-eager logical TP2 oracle is not vLLM Marlin arithmetic.
- vLLM `0.28.0+cu129` / Torch `2.13.0+cu129`: eager, Marlin FP8 linear/MoE; logs confirm Triton/FLA GDN prefill and Triton GDN decode fallback. Fixed `NCCL_P2P_DISABLE=1`, NCCL 2.29.7+CUDA12.9 and `SHM/direct/direct` observed. No silent transport override.
- Ferrule CUDA12.4 allowlist and `CUBLAS_WORKSPACE_CONFIG=:0:0` stay separate from vLLM CUDA12.9 JIT allowlist. vLLM process maps additionally contain CUDA13.2 runtime/NVRTC, as in the historical baseline; preserve this dynamic-loader closure caveat. No library was changed.
- Ferrule predeclared KV16 MiB / workspace1024 MiB / max-tensor2048 MiB is not vLLM KV allocation parity. vLLM planning remains 19.2 GiB/card; common sampled ceiling is 22 GiB/card plus 128 GiB PSS, not physical partitions.
- Performance slots use `FERRULE_LOG=info`, no request DEBUG, zero numerical/state trace reserves; vLLM detailed traces/NVTX disabled. Source/profile/log evidence is not full per-operation arithmetic observation.
- All original visible canaries match manifest `102c798baeb5200bc8ed8386eabed6146870b3b6da2a90fc6010677041352c7e`. HTTP text/usage is not observed token IDs/logits/routes/state. Model/runtime numerical proof remains distinct from release HTTP evidence.

## Custody, memory and failure denominators

- Six actual starts: two smoke + four paired, all complete. Paired measured requests: 20/20 successful; all paired HTTP requests: 48/48. Smoke: 20 normal requests + two verified intentional cancellations. No startup/request/OOM/timeout/unknown failure in this pilot.
- Every fresh lifecycle used sorted atomic two-UUID locks and the existing sealed authenticated subreaper; root PID/start-time custody, no foreign signals, pre/post all-device idle and final ECHILD drain. Each lifecycle retained startup900/request120/all-requests440/teardown120, supervisor1600 and ceiling1700 seconds.
- Pre-launch and final memory were zero on all eight devices; all nonselected sampled deltas were zero. Explicit tolerance remains 1 MiB. All six lifecycles had zero missing GPU attempts and zero incomplete PSS samples, but sampled peaks are not exact/continuous memory bounds.
- Independent final lock reacquisition verified all eight idle/no compute apps and all six old server identities gone, then released both UUID locks. NVML host-PID mapping/owned GPU attribution remains unknown.
- Memory peaks cover loading and requests, not only steady state. OS/JIT/page caches were uncontrolled; fresh servers are not storage-cold. Historical 11.31 tok/s baseline and M7/G10/failed canary/startup receipts remain unchanged and are not spliced into this pair.

## Receipts and validation

- [Machine-readable summary, per-request populations, memory/delta evidence and artifact pins](runs/tp2-fv-qa8-20261009-001/serving-pilot-summary.json)
- [Frozen paired preparation](runs/tp2-fv-qa8-20261009-001/prepared.json) · [Execution](runs/tp2-fv-qa8-20261009-001/execution.json)
- [Same-release smoke receipt](runs/tp2-serving-smoke-20261009-003/serving-smoke.json)
- [Final all-eight idle / server identities](runs/tp2-fv-qa8-20261009-001/final-owner-recheck.json) · [Release](runs/tp2-fv-qa8-20261009-001/final-owner-release.json)
- Logs and HTTP evidence: `runs/tp2-fv-qa8-20261009-001/runs/{F1,V1,V2,F2}/session/`.
- CPU validation: **42 original + 78 TP2 tests passed**. Source/runtime/reference/smoke/pair pins rechecked; historical run files rehashed. No production/oracle/root-script edits.

| Slot | Authenticated launch ID |
|---|---|
| F1 | `232ee94989854072bd248163c89bb964` |
| V1 | `113836a0f083424a8923c3f3724a4ffa` |
| V2 | `faa1f1c6a50d4413a6a3715a9f9e1887` |
| F2 | `ca030a5d73df4f348e6c07b49211b5bd` |
| S1 | `de8445b93b7047cd8e5a02143810e5ac` |
| S2 | `e1c9d4155c464b79b429fe7ccab747d4` |

Next work requires a separately reviewed protocol/authorization: investigate the Ferrule startup/PSS/request-time gaps without attributing them solely to collectives, batching or arithmetic from this pilot. Do not reuse the consumed run IDs or mutate their pins.
