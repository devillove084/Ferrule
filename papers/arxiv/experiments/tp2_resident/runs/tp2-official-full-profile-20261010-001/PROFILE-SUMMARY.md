# Full-model official Nsight Systems profile

## Verdict

**REJECTED_INCOMPLETE_TRACE** — not a complete profile PASS and not a benchmark.

The existing paired/collector-first supervised lifecycle completed once with the fixed qualified Ferrule binary and the official NVIDIA Nsight Systems 2025.3.1 CLI/importer/CUPTI set. Both owner request windows were matched, but the official SQLite diagnostics include `Not all CUDA events might have been collected`. The CPU-only v2 audit finds 16 warmup activity rows missing an API (4 memcpy + 12 memset), plus two successful `cuLaunchKernel` calls missing activity in the entire export: correlation IDs **687294 / 687352**, API start **816884800 / 816930550 ns**. Their early-window location is observed; **capture-boundary causation is unproven**. The warning comes from the generic `libAnalysis` TraceSummaryVerifier completeness check, not solely CUDA-event dependency tracing. No explicit CUPTI-dropped diagnostic establishes buffer overflow; the cause remains unknown. Raw and old receipts are retained; no second full-model run was attempted.

## Fixed scope

- Binary SHA256: `4d40c34738d36a1c78f62c0e400dd732da40e34c316ce4bd3c356dc4d29b3173`
- TP2 thread, PP1, EP1; GPU0/1 only:
  - `GPU-b980f22b-58a3-fc88-21c1-ae86053d8535`
  - `GPU-63f1bd01-ddf9-a742-ee48-9389892493d9`
- Context 128, batch tokens 8, max active sequences 2, 22 GiB/card, `CUBLAS_WORKSPACE_CONFIG=:0:0`.
- Official Nsight Systems `2025.3.1.90-253135822126v0`, matching importer, bundled `libcupti.so.12.9`.
- Existing paired lifecycle only; no second server harness. Startup/readiness and Hello/canary occurred before capture. Capture window contained one warmup and one measured QA8 request.

## Lifecycle evidence

- `execution.json`: `complete_profile`
- `server_starts`: 1
- `authenticated_drained`: true
- `after`: `verified`
- Eight instrumentation locks: released; original B0 verified.
- Final host check: all eight GPUs 0 MiB / 0%, no compute process.
- HTTP warmup and measured requests both passed, each 5 prompt tokens and 8 generated tokens.
- Profile time is diagnostic only; no performance comparison or benchmark claim is made.

## Matched request windows

Request IDs came from `ferrule_request` admission/terminal markers and were aligned to Nsight's `TARGET_INFO_SESSION_START_TIME`.

- Warmup: request 5; `1,887.967 ms` wall window; 87,310 kernels, 20,400 memcpy, 87,310 kernels correlated, 20,396 memcpy correlated. Both owner UUIDs present.
- Measured: request 6, **[2622335777, 5112115777] ns**, `2,489.780 ms` wall window; **117,232 kernels, 26,284 memcpy, 70,014 memset, 1,881,862 APIs**. **115,380 runtime + 1,852 driver launches = 117,232 kernels**. Both correlation directions are unique, counterparts remain within this request (no startup/cross-window match), and same-thread API overlap count is zero. Both owner UUIDs are correct. Status: **`observed_correlation_closed`**.

Warmup has 16 forward and two reverse missing correlations. Request6's exported-row closure does **not** exclude paired API/activity loss. Both request and report `full_trace_accepted` remain **false**; no complete-collection, performance, or E2 claim follows.

## Quantified local evidence

Measured request only, summed over both owner GPUs and exact kernel names:

- `decode_weights`: **26,592 instances / 270507218 ns** (270.507 ms) GPU kernel time.
- `pack_bf16_activation`: **19,228 instances / 30012786 ns** (30.013 ms).
- Matrix-math names classified conservatively as not a cuBLAS API measurement: **24,682 / 129.517 ms**.
- `bf16_gemmex::split_pair`: **7,364 / 11.965 ms**; this is a kernel name, not `cublasGemmEx` API timing.
- Memcpy bytes: H2D 21.882 MB, D2D 643.400 MB, D2H 40.868 MB.
- CUDA synchronization API evidence includes `cuEventSynchronize`, `cuStreamSynchronize`, `cuEventQuery`, and `cuStreamWaitEvent`; `cuEventQuery` return 600 is treated as CUDA-not-ready polling, not a fatal execution error.
- Launch gaps/inter-arrival distributions and per-UUID kernel/memory rows are in `profile-audit-v2.json`; v1 is preserved unchanged.
- No exact prefill/decode kernel marker exists. “Before first emission” and “after first emission” are retained only as explicit proxy partitions; no kernel is falsely attributed to prefill or decode.
- No direct CPU collective marker was found. OSRT waits and launch gaps are retained, but are not labeled as collective waits.
- No `cublasGemmEx` API table/range was exported. It is explicitly reported as unavailable rather than inferred from kernel names.

## Candidate optimization directions (not proven wins)

1. **Reduce repeated weight dequantization and activation packing.** `decode_weights` + `pack_bf16_activation` contribute **300.520 ms / 47.63%** of the measured request's **630.996 ms** observed kernel-duration sum. Consider fused FP8 consumption or selective reuse within the 22 GiB/card ceiling, not unrestricted decoded-weight caching. Validate numerical equivalence and remeasure with a complete trace before claiming a speedup.
2. **Amortize event lifecycle and retirement checks.** The measured window has **270,580 creates**, **270,580 records**, **269,848 destroys**, and **260,563 queries**: **1,071,571 calls / 459.732 ms** host API duration in total. `cuEventSynchronize` adds **17,050 calls / 194.060 ms**; `cuStreamSynchronize` adds **67,866 / 82.988 ms**. Investigate event reuse/batched completion while retaining ownership and retirement correctness. These are overlapping host-duration sums including GPU waits, not additive speedup estimates.
3. **Investigate small host round trips and submission granularity.** The measured window has **3,324 D2H / 40,867,912 B** and **14,244 H2D / 21,881,944 B**. D2H GPU transfer time is **6.453 ms**, versus **1,004.509 ms** in the host D2H API (maximum **467.274 ms**); H2D host API time is **74.349 ms**. `cudaLaunchKernel` contributes **115,380 calls / 492.516 ms**. Consider batching host exchanges or retaining eligible intermediates on device. The large stall may include profiler effects/driver waits: do not label it pure copy bandwidth or CPU collective cost. Owner-thread futex totals are **106.515 ms** (GPU0 launcher TID167636) and **122.583 ms** (GPU1 launcher TID167637), but lack collective callsite attribution.

All three are hypotheses backed by observed local quantities, not accepted full-trace bottleneck rankings or demonstrated gains. Cross-thread/device sums overlap. Missing events can inflate observed launch/idle gaps.

## W8 quality constraint

The default-off W8 routed-down candidate **failed the real-operand exact gate**:
16382/16384 words match versus native's 16384/16384; two rank1 words differ.
It is not a qualified replacement for the above dequantization/pack work and
remains default-off. See [model test result](../../../../../../crates/ferrule-model/tests/mixed_routed_down/RESULT.md).
This paper-only closeout changes no model/kernel, expectation or threshold.

## Artifacts

- Raw report: `runs/P1/session/nsight/nsys.nsys-rep`
- Exported SQLite: `runs/P1/session/nsight/nsys.sqlite`
- NVIDIA summaries: `runs/P1/session/nsight/cuda_api_sum.txt`, `cuda_gpu_kern_sum.txt`, `cuda_gpu_mem_size_sum.txt`, `cuda_gpu_mem_time_sum.txt`, `osrt_sum.txt`, `nvtx_sum.txt`
- Revised bidirectional audit: `profile-audit-v2.json`
- Preserved original audit: `profile-audit.json`
- Original summary/script snapshots: `cpu-closeout-v2/PROFILE-SUMMARY-v1.md`, `cpu-closeout-v2/profile_trace_audit_v1.py`
- Original receipt manifests remain unchanged; the CPU closeout has a separate manifest.
- Request receipts: `runs/P1/session/profile/warmup.json`, `measured.json`, `profile-window.json`
- Request/server marker log: `runs/P1/session/server.log`
- Lifecycle receipts: `execution.json`, `runs/P1/session/supervisor.json`, `runs/P1/probe-lock-release.json`, `runs/P1/lock-release.json`

The raw report and SQLite are preserved unchanged after collection. No full-model retry was run.
