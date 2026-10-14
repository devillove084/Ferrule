# Storage-demand TP1 middle: offline receipt tables

Single instrumented successful run, n=1; descriptive observations only. No throughput comparison, statistics, or confidence intervals.

## Attempt accounting

| Attempt | Status | Planned | Dispatched | Success | Not run | Usage output tokens | Invocations |
|---|---|---:|---:|---:|---:|---:|---:|
| -01 | failed | 4 | 0 | 0 | 4 | 0 | 0 |
| -02 | complete | 4 | 4 | 4 | 0 | 26 | 30 |

-01 failed before server launch; its unobserved counters remain null, not zero.

## Per-request latency (-02)

| Request | Input / output usage tokens | Invocations | TTFT (s) | E2E (s) | Content-event / usage TPOT proxy (s) |
|---|---:|---:|---:|---:|---:|
| hello2 | 1 / 2 | 3 | 4.265 | 9.186 | 3.114 |
| canary-qa | 5 / 8 | 9 | 8.332 | 18.309 | 1.285 |
| canary-arithmetic | 5 / 8 | 9 | 5.308 | 14.110 | 1.072 |
| canary-chinese | 3 / 8 | 9 | 2.605 | 9.932 | 0.884 |

Launch-to-ready: 18.497 s. Latency follows the HTTP receipt, not exact token or device-kernel time. No generated IDs were observed.

Hello2: HTTP completion and usage checks only; no reference comparison. Only 3 canaries have visible-text/eight-output-usage reference checks; HTTP success is not a frozen-reference PASS. No exact generated IDs or numeric acceptance.

## Logical additive invocation counters (-02)

| Request | Read calls | Reader bytes | H2D copies | H2D bytes | Artifact uploads | Expert hits / uploads / evictions |
|---|---:|---:|---:|---:|---:|---:|
| hello2 | 5,760 | 3,020,267,520 | 7,533 | 3,020,281,452 | 2,880 | 0 / 960 / 896 |
| canary-qa | 22,068 | 11,571,399,936 | 28,185 | 11,571,459,828 | 11,034 | 0 / 3,678 / 3,678 |
| canary-arithmetic | 21,288 | 11,162,405,376 | 27,275 | 11,162,465,268 | 10,644 | 0 / 3,548 / 3,548 |
| canary-chinese | 19,998 | 10,485,991,296 | 25,770 | 10,486,042,140 | 9,999 | 0 / 3,333 / 3,333 |
| Total (30 invocations) | 69,114 | 36,240,064,128 | 88,763 | 36,240,248,688 | 34,557 | 0 / 11,519 / 11,455 |

Reader bytes are logical checkpoint demand, not physical NAS/NFS IO. Facade H2D is not physical bus traffic. Uploads are not cache misses.

## Cache gauges (-02; never summed)

| Gauge | Last observed invocation | Observed maximum |
|---|---:|---:|
| `expert_resident_entries` | 64 | 64 |
| `expert_resident_bytes` | 201,351,168 | 201,351,168 |
| `expert_pending_upload_bytes` | 0 | 0 |
| `expert_scratch_bytes` | 8,388,608 | 8,388,608 |
| `expert_owner_peak_bytes` | 219,754,496 | 219,754,496 |
| `expert_unknown_quarantine_bytes` | 0 | 0 |

These are invocation observations, not a final shutdown ledger. Byte gauges are bytes; resident entries are a count.

## Sampled memory (-02; bytes)

| Phase | Samples | Selected-device peak | Tree PSS peak | Tree RSS peak | Tree Anonymous peak |
|---|---:|---:|---:|---:|---:|
| All samples | 32 | 4,325,376,000 | 7,980,569,600 | 7,983,616,000 | 7,818,027,008 |
| startup | 9 | 2,905,604,096 | 7,980,569,600 | 7,983,616,000 | 7,818,027,008 |
| ready_before_first | 0 | — | — | — | — |
| first_request | 4 | 4,325,376,000 | 6,975,266,816 | 6,978,363,392 | 6,808,907,776 |
| after_first_request | 19 | 4,325,376,000 | 6,979,462,144 | 6,982,508,544 | 6,812,921,856 |

Independent sampled maxima, not additive components or exact peaks. Ready-device snapshot: 3,989,831,680 bytes. PssAnon/PssFile and physical host cap are unknown. After-first-request is not steady state.

resource_contract_complete = unknown for both attempts. Full caveats/provenance: `REPORT.md` and `summary.json`.

**Source consistency: PASS**.
