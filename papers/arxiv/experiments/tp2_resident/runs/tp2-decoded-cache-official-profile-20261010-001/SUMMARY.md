# Cache-on strict TP2: completed execution and observed official trace

## Conclusion

**Observed formal-request decode_weights and activation-pack counts/durations decreased. End-to-end throughput did not improve in this pilot.**
Both new trace windows have bidirectional observed correlation closure, but the official completeness warning remains: **full_trace_accepted=false**. These are qualified local observations, not a complete-trace, causal cache-only speedup, or arithmetic-parity claim.

Final binary SHA256: `dadf831eb2af477108efd402171c40f7544a2600d14b8e5e7ab8a456d2625e5a`.
Final -03 v2 handoff SHA256: `90a8289e18334aaae4dde93f0ae763868faf3b78e666258fddedd96e6e7b09fd`.

Executed once in the requested order: same-release S1/S2 smoke, no-trace F1/V1/V2/F2, then one independent official Nsight 2025.3.1 profile. All seven lifecycles passed their existing authenticated supervisor/drain and resource/request checks. No production edits, runner patches, retries, tool installs, or foreign process signals were needed.

## No-trace HTTP benchmark

Each slot used excluded first QA8 + two warmups + five measured requests; concurrency 1, five input tokens and eight output tokens per measured request. All 20 measured requests / 160 output tokens passed. Throughput uses first measured send to last client terminal, including client overhead; TTFT/E2E below are means. Smoke and profile timings are excluded.

| Slot | Startup s | Mean TTFT ms | Mean E2E ms | tok/s | Sampled GPU peak GiB/card | Tree PSS peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| F1 | 351.694 | 525.993 | 1458.338 | 5.3888 | 19.667 | 60.759 |
| V1 | 67.849 | 132.976 | 707.479 | 10.1374 | 19.284 | 8.349 |
| V2 | 62.327 | 129.827 | 686.410 | 10.5434 | 19.284 | 7.911 |
| F2 | 351.611 | 519.826 | 1434.800 | 5.4275 | 19.667 | 59.081 |

### Historical cache-off comparison: descriptive only

Cache-off `4d40c347…b3173` used workspace 1024 MiB; cache-on used **4096 MiB**. Both kept a 22 GiB/card total cap. The pinned vLLM TP2/EP-disabled Marlin baseline metadata is identical. CPU/host TP2 collectives versus NCCL SHM and Mixed versus Marlin remain non-equivalent arithmetic implementations. OS/JIT/page caches and time-of-run effects are uncontrolled.

| Ferrule slot | Cache-off startup s → cache-on | Cache-off tok/s → cache-on | Descriptive tok/s change |
|---|---:|---:|---:|
| F1 | 243.480 → 351.694 | 5.5483 → 5.3888 | -2.88% |
| F2 | 241.209 → 351.611 | 5.5586 → 5.4275 | -2.36% |

Do not call these differences a controlled regression or speedup. `fairmatch=false`, `exact_speedup=null`; no statistical significance, throughput parity, or universal E2 acceptance is claimed.

## Cache physical and activity observations

S1/S2 startup was 361.901 / 363.621 s; both smoke lifecycles passed visible canaries, cross-request reuse, cancellation/post-cancel reuse, normal shutdown and rebuilt owner identity. These diagnostic timings are not benchmark results.

Both owners, across smoke/pair/profile, had ready current **2480 MiB**, ready peak **2543 MiB**, shutdown current **3010 MiB**, and shutdown peak **3072 MiB**. Current component sums and peak <=3 GiB were checked independently of whole-device GPU sampling.

Profile shutdown physical ledger per owner:

| Component | MiB |
|---|---:|
| `pinned_grants` | 1340 |
| `active_routed_grants` | 281 |
| `pooled_grants` | 0 |
| `pending_reserved_grants` | 0 |
| `segment_slack_bytes` | 1389 |
| `pending_segment_slack_bytes` | 0 |
| `physical_current_bytes` | 3010 |
| `physical_peak_bytes` | 3072 |
| `driver_allocation_growth_bytes` | 1600 |

`driver_allocation_growth_bytes` is an observation, **not an additional component charge**. The large segment-slack reserve is conservative cache accounting, not necessarily physically wasted device memory. Unknown reservations, borrowed keys and pending allocations/leases were zero at ready/shutdown; both owners were published, idle and not quarantined. The snapshot is a host-side cache ledger, not a CUDA completion receipt.

Ordinary smoke/benchmark rankperf stayed **off**: hits/misses/decoded activity are **null (uncollected), not zero**. Each F1/F2 owner recorded 81,328 lifetime backing reuses/evictions and 435 native fallbacks, which are available without rankperf.

Profile-only `FERRULE_LOG=info,ferrule_request=debug,ferrule_performance=debug` enabled the runtime setter and model logging gate. The resulting overhead is disclosed and profile timings must not enter the benchmark table. The source subscriber reads FERRULE_LOG, not RUST_LOG.

| Owner | Profiled invocations | Hits | Misses | Hit fraction | Logical BF16 returned bytes | Evictions | Native fallbacks |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 48 | 12,000 | 37,766 | 24.113% | 59,191,066,624 | 37,379 | 200 |
| 1 | 48 | 12,000 | 37,766 | 24.113% | 59,191,066,624 | 37,379 | 200 |

These counters cover the **whole enabled profile lifecycle**, including Hello/canaries outside the CUDA capture; they are not formal-request-only counts. Miss semantics include successful decode misses and native fallbacks. Logical returned BF16 bytes are neither new H2D uploads nor GPU-completed bytes. The 96 invocation summaries exactly agree with both shutdown activity snapshots; 3,936 layer/invocation debug records were retained. Cache churn remains high, but these observations alone do not causally explain throughput.

## Official Nsight formal request6: same workload, local observed comparison

Both official runs used the same fixed QA8 warmup/measured workload: five input tokens, eight output tokens, identical visible text and `length` finish. The measured window is bound to the HTTP/SSE request ID, server admission/terminal markers and official session epoch. New window: **[3035275306, 5624168306] ns**. New warmup window: [740412306, 2741809306] ns. These durations are profiling observations only.

| Kernel category | Cache-off count → cache-on | Count change | Cache-off sum ms → cache-on | Sum-duration change |
|---|---:|---:|---:|---:|
| `decode_weights` | 26,592 → 22,092 | -16.92% | 270.507218 → 149.980197 | -44.56% |
| `pack_bf16_activation` | 19,228 → 78 | -99.59% | 30.012786 → 0.123969 | -99.59% |

These are observed reductions, conditional on exported-row coverage. Kernel sums overlap across owners and are not wall time. Different workspace and new profile-only rankperf logging preclude an identical-deployment causal comparison. Pure prefill/decode boundaries remain unavailable; first-emission splits are explicit proxies only. Matrix-named kernels are not measured GemmEx API ranges.

### Launch/API/event observations within the formal window

| Observation | Cache-off | Cache-on |
|---|---:|---:|
| `kernel_observed` | 117,232 | 93,582 |
| `launch_runtime` | 115,380 | 91,730 |
| `launch_driver` | 1,852 | 1,852 |
| `runtime_api_rows` | 1,881,862 | 1,857,858 |
| `memcpy_observed` | 26,284 | 26,284 |
| `memset_observed` | 70,014 | 70,014 |

| API | Cache-off count / host sum ms | Cache-on count / host sum ms |
|---|---:|---:|
| `cuEventCreate` | 270,580 / 210.766636 | 270,580 / 238.297814 |
| `cuEventRecord` | 270,580 / 77.071449 | 270,580 / 85.895608 |
| `cuEventDestroy_v2` | 269,848 / 70.162075 | 269,848 / 104.634466 |
| `cuEventQuery` | 260,563 / 101.731819 | 260,386 / 123.737183 |
| `cuEventSynchronize` | 17,050 / 194.059671 | 17,050 / 103.968645 |
| `cuStreamSynchronize` | 67,866 / 82.987683 | 67,866 / 83.300323 |
| `cuStreamWaitEvent` | 408 / 0.224425 | 408 / 0.218332 |
| `cudaLaunchKernel_v7000` | 115,380 / 492.516429 | 91,730 / 402.672942 |
| `cuLaunchKernel` | 1,852 / 7.833334 | 1,852 / 7.758154 |

Host API durations include GPU waits, overlap across threads and contain profiler perturbation. Event create/record/destroy and stream/event synchronization counts did not fall. `cuEventQuery` return 600 is not-ready polling, not an execution failure.

### DMA within the formal window

| Direction | Copies, old = new | Bytes, old = new | Old sum ms | New sum ms |
|---|---:|---:|---:|---:|
| H2D | 14,244 | 21,881,944 | 8.319528 | 8.313030 |
| D2D | 8,716 | 643,399,680 | 12.574283 | 12.557008 |
| D2H | 3,324 | 40,867,912 | 6.452807 | 6.495014 |

No observed DMA count/byte reduction is demonstrated. Aggregate transfer/kernel/API time is not an additive request-time decomposition.

### Trace acceptance limitations

- Both new warmup and formal request6 are `observed_correlation_closed`: unique successful activity→API and API→activity matches, full-export PID/correlation search, same-window containment, correct two GPU UUIDs and zero same-thread runtime overlap.
- Historical cache-off formal request6 is also closed; its warmup retained missing activity/API records. Historical receipts were not changed.
- The new official importer still reports **“Not all CUDA events might have been collected.”** Additional NVTX/OSRT/scheduling warnings remain in the raw diagnostics. No unsupported cause is assigned.
- Unique exported-row closure cannot exclude paired API/activity loss. **`full_trace_accepted=false` remains mandatory**, even though new warmup is closed.
- No exact prefill/decode kernel attribution, GemmEx API measurement or CPU-collective callsite decomposition is claimed. No second capture was performed.

## Custody, release and validation

Profile used all eight sorted UUID locks for instrumentation, original all-zero B0, stable collector-only B1 (4 MiB/selected card and 1 MiB/nonselected card), model visibility restricted to ordered GPU0/1, return to B1 after model reap, then collector drain back to original B0.

Independent final verification: **all eight GPUs 0 bytes / 0% utilization, no compute PID; all eight existing UUID locks reacquired nonblocking and released; all seven supervisors ECHILD/drained; 24 recorded owned process identities absent.** Final source/admission/tool pins revalidated. No foreign signal was sent.

CPU closeout: **215 tests PASS**, with CUDA hidden. Numerical test-binary qualification remains separate from public HTTP visible-canary acceptance. No production source or historical receipt was modified.

## Evidence

- [Machine-readable measured results and input hashes](execution-summary.json)
- [Official bidirectional audit v2](profile-audit-v2.json)
- [Final idle, process and lock release](final-owner-release.json)
- [CPU test receipt](cpu-closeout-tests.json)
- [Same-binary smoke](../tp2-decoded-cache-smoke-20261010-001/serving-smoke.json)
- [No-trace pair execution](../tp2-decoded-cache-fv-20261010-001/execution.json)
- [Official raw report](runs/P1/session/nsight/nsys.nsys-rep)
- [Official SQLite export](runs/P1/session/nsight/nsys.sqlite)
- [Official GPU kernel summary](runs/P1/session/nsight/cuda_gpu_kern_sum.txt)
- [Official CUDA API summary](runs/P1/session/nsight/cuda_api_sum.txt)
- [Profile server log with cache snapshots](runs/P1/session/server.log)
