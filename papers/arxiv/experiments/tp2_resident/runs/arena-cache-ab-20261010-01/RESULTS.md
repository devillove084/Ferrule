# Arena same-binary smoke and cache A/B

Actual S1/S2 smoke PASS for `ae3e76d4…731bc7`; actual CPU promote completed.
OFF `4d40c347…9b3173` and ON `ae3e76d4…731bc7` each ran one fresh lifecycle.
No vLLM or Nsight run; no production/model changes.

Both: TP2, ctx128, batch8, seq2, resident/no offload; workspace4096 MiB, total22528 MiB/card.
QA8: excluded first + 2 warmups (3 unmeasured), then 5 measured; all canaries and 5/5 measured passed.

| Metric | OFF | ON |
|---|---:|---:|
| Output tokens/s (usage-based) | 5.3257 | 2.5071 |
| Requests/s | 0.6657 | 0.3134 |
| Startup seconds | 245.1736 | 347.8984 |
| Host tree PSS peak GiB | 49.9147 | 60.1439 |
| TTFT mean / p50 / p95 seconds | 0.5256 / 0.5251 / 0.5278 | 1.3645 / 1.3453 / 1.4165 |
| Whole-card sampled peak GiB (each selected card) | 18.04199 | 21.04199 |

ON/OFF output-rate ratio: 0.4708; observed ON throughput was 52.92% lower in this run, not a statistically significant effect claim. Startup excluded from throughput.

## Arena and custody

All ON/S1/S2 snapshots passed v2 physical partition validation: each owner current/peak/root/driver growth = 3 GiB; unknown/quarantine absent.
Ready: pinned 1340 MiB + free 1732 MiB = 3072 MiB. Shutdown: pinned 1340 MiB + active routed 1732 MiB = 3072 MiB.
ON shutdown per owner: backing_reuses=54621, evictions=54621, native_fallbacks=0. S1/S2 each: reuses=46330, evictions=46330 per owner.
These are entire-lifecycle counters, not measured-phase counters. New allocation count is unavailable; growth bytes must not be relabelled as an allocation count.
Rankperf remained off; activity=null, not zero decode. No hot zero-decode/no-churn conclusion.

All four launches have distinct launch IDs, selected-roster locks, pre/post idle, authenticated successful supervisors, ECHILD and drain.
Raw eight-GPU samples stayed <=22 GiB per card. Final eight GPUs are zero-used/zero-utilization idle; selected locks released.
No foreign process was signalled. NVML/local PID attribution remains unknown: no proof of absence of foreign interference beyond the exclusive-owner assumption and monitored baseline contract.

## Limits and evidence

One OFF then one ON lifecycle, n=5 measured each; same flags but different sources, **not pure cache effect or statistical significance**.
Sampled maxima are not continuous guarantees. Four native-CPU qualification source-pin gaps remain disclosed.
`owner-closeout.json` links full smoke/promotion/A-B evidence, raw sample hashes, lifecycle snapshots and final idle/release.
`cache-ab-summary.json` retains original unmodified benchmark results; `final-idle.json` and `final-lock-release.json` record final handoff.
