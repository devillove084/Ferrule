# Indexed arena crossed A/B — actual GPU closeout

Same-binary 9ef S1/S2 PASS, actual promote completed, then two crossed cycles PASS. No Nsight/vLLM or production changes. CPU suite 280/280 PASS.

BEFORE `ae3e76d420cdff234be4262575b3bee29e89252c89bc2f21c744aefd47731bc7`; AFTER `9ef9841598b4694fb611250bf8876c4314d252400136ddb23f606a979cadb656`.

Both: arena3GiB, workspace4096MiB, device22528MiB/card; TP2 ctx128 batch8 seq2 resident/no-offload; concurrency1. Each lifecycle: 3 unmeasured QA8 (excluded first +2 warmups), then 5 measured with input5/output8 and frozen canaries.

| Cycle/order | Variant | Startup s | TTFT mean ms | E2E mean ms | Output tok/s | GPU peak GiB/card | Host PSS peak GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 / BEFORE then AFTER | BEFORE | 355.329678 | 1421.311172 | 3261.790039 | 2.422126 | 21.041992 | 61.771711 |
| 1 / BEFORE then AFTER | AFTER | 358.708880 | 540.415555 | 1483.723826 | 5.205591 | 21.041992 | 61.221485 |
| 2 / AFTER then BEFORE | AFTER | 356.955392 | 537.151549 | 1466.656233 | 5.247737 | 21.041992 | 61.090178 |
| 2 / AFTER then BEFORE | BEFORE | 348.405517 | 1376.675528 | 3170.049223 | 2.485096 | 21.041992 | 61.084245 |

Rates use the original actual runner-observed run window, including terminal checkpoint/executor exit; no alternate rate is substituted. Full p50/p95, run windows/tails and raw hashes are in owner-closeout.json. Startup is excluded from measured throughput.

- Cycle 1 descriptive AFTER/BEFORE output-rate ratio: 2.149183.
- Cycle 2 descriptive AFTER/BEFORE output-rate ratio: 2.111683.

## Per-rank arena counters and cleanup

Every ready/shutdown snapshot validates schema v2 with root/current/peak/driver growth 3072 MiB, zero Unknown/quarantine. Ready partitions: pinned1340 + free1732 MiB. Shutdown partitions: pinned1340 + active routed1732 MiB. Root is counted once, not added again to slot partitions.

- smoke S1: rank0/rank1 shutdown counters `[{'backing_reuses': 46330, 'evictions': 46330, 'native_fallbacks': 0}, {'backing_reuses': 46330, 'evictions': 46330, 'native_fallbacks': 0}]`; GPU samples 157, missing 0; monitor_terminal excluded.
- smoke S2: rank0/rank1 shutdown counters `[{'backing_reuses': 46330, 'evictions': 46330, 'native_fallbacks': 0}, {'backing_reuses': 46330, 'evictions': 46330, 'native_fallbacks': 0}]`; GPU samples 154, missing 0; monitor_terminal excluded.
- cycle1 BEFORE: rank0/rank1 shutdown counters `[{'backing_reuses': 54621, 'evictions': 54621, 'native_fallbacks': 0}, {'backing_reuses': 54621, 'evictions': 54621, 'native_fallbacks': 0}]`; GPU samples 164, missing 0; monitor_terminal excluded.
- cycle1 AFTER: rank0/rank1 shutdown counters `[{'backing_reuses': 54621, 'evictions': 54621, 'native_fallbacks': 0}, {'backing_reuses': 54621, 'evictions': 54621, 'native_fallbacks': 0}]`; GPU samples 160, missing 0; monitor_terminal excluded.
- cycle2 AFTER: rank0/rank1 shutdown counters `[{'backing_reuses': 54621, 'evictions': 54621, 'native_fallbacks': 0}, {'backing_reuses': 54621, 'evictions': 54621, 'native_fallbacks': 0}]`; GPU samples 158, missing 0; monitor_terminal excluded.
- cycle2 BEFORE: rank0/rank1 shutdown counters `[{'backing_reuses': 54621, 'evictions': 54621, 'native_fallbacks': 0}, {'backing_reuses': 54621, 'evictions': 54621, 'native_fallbacks': 0}]`; GPU samples 161, missing 0; monitor_terminal excluded.

The four A/B lifecycles report equal 54,621 reuses/evictions per rank; this equality is an observation, not a forced route assertion or hot-phase count. Activity remains null. New allocation count is not exported; driver-growth bytes are not an allocation count.

Six distinct owner launches verified: same selected UUID roster/locks, pre/post idle, authenticated bounded supervisors, ECHILD/drain and owned process identities no longer live. Raw eight-GPU samples all satisfy <=22GiB/card. Captured no-trace command/environment matches across all four A/B lifecycles (`FERRULE_LOG=info`). Final eight GPUs zero-used/zero-utilization, selected locks released, no foreign signals.

## Limits

- Two crossed cycles strengthen the descriptive index comparison versus the previous 4d40/cacheoff comparison, but n=2 cycles is not statistical acceptance.
- Only two production source files differ; build/tool/runtime flags match. This does not attribute every timing difference exclusively to indexing.
- Evictions/backing reuses are cumulative across whole lifecycles including canaries/warmups. No route samples or measured-phase cache activity exist; equality of counters does not prove equal routes.
- Rankperf off and activity null; no zero-decode/no-DMA inference. No Nsight or vLLM execution.
- NVML host-to-local PID mapping unavailable; local authenticated drain is proven, absence of foreign interference is not. Explicit sole-owner allocation is an assumption.
- GPU peaks are sampled, not continuous guarantees. CPU affinity/thread count and GPU temperature/clocks/power are uncollected, so no oversubscription/thermal attribution.
- Four native CPU .cc qualification-source coverage gaps remain disclosed; they are build-pinned and unchanged, not retroactively GPU-source-covered.

See owner-closeout.json for exact binary/release/delta/source identities, raw report pins and final idle/release links.
