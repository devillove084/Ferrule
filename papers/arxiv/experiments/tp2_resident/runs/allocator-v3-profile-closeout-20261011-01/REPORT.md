# Official allocator V3 profile — partial trace, formal windows observed-closed

Exactly two existing prepared lifecycles executed once: BEFORE 9ef and AFTER 7e1c. Official Nsight 2025.3.1 CLI/Importer fixed hashes and loaded CUPTI 12.9 verified. Same actual text: ` Tokyo.\nThe capital of Japan is`; both formal requests 5 input / 8 output tokens. Existing actual smoke reused, no new smoke or benchmark.

## Acceptance boundary

Both formal HTTP-anchor windows have unique successful bidirectional kernel/memcpy/memset↔API correlation, actual model PID and selected physical UUID mapping. BEFORE warmup is also closed; AFTER warmup is NOT closed: 20 memcpy and 53 memset rows lack APIs, and 4 successful launch APIs lack kernels. All 77 issues and all Nsight diagnostics are preserved. The model warning `Not all CUDA events might have been collected` remains; exported closure cannot exclude simultaneous API/activity loss. **full_trace_accepted=false**. No blind rerun.

## Formal request API observations

Host durations below sum exported API intervals across both rank threads. They are not wall time, GPU time, or additive savings. Driver cu* calls are stored in this Nsight RUNTIME table; absent separate DRIVER table is null, not zero driver calls. No cudaEvent* runtime-named calls were observed in these formal exports.

| API | BEFORE count | BEFORE host ms | AFTER count | AFTER host ms |
|---|---:|---:|---:|---:|
| cuEvent create | 270584 | 217.193 | 75218 | 56.968 |
| cuEvent record | 270584 | 80.943 | 75218 | 26.743 |
| cuEvent query | 261676 | 98.194 | 62270 | 29.619 |
| cuEvent destroy | 270584 | 72.660 | 75218 | 21.207 |
| cuEventSynchronize | 17054 | 114.144 | 17054 | 136.972 |
| cuStreamSynchronize | 67866 | 84.481 | 67866 | 82.363 |
| cuStreamWaitEvent | 408 | 0.205 | 408 | 0.225 |
| cudaLaunchKernel_v7000 | 84540 | 354.854 | 84540 | 399.498 |
| cuLaunchKernel | 1852 | 8.027 | 1852 | 8.065 |

Four event API families total: 1,073,428 / 468.991 ms → 287,924 / 134.537 ms. These are API calls, not unique events/completed retirements. Query not-ready return 600: 8,110 → 4,070; not numerical failure. Full return-code distributions are in the JSON reports.

## What still dominates observed kernels

Both variants have **86,392 kernels = 84,540 runtime + 1,852 driver launch calls**, 43,196 kernels per GPU. Kernel name + grid/block dimensions + count per physical UUID match exactly. This proves observed launch-shape equality, not tensor-pointer/argument identity. No pure prefill/decode split is available.

| Kernel group | Count, both variants | BEFORE summed GPU ms | AFTER summed GPU ms |
|---|---:|---:|---:|
| decode_weights | 14980 | 101.607 | 101.634 |
| pack_bf16_activation | 0 | 0.000 | 0.000 |
| gemm_gemv_named | 24682 | 135.180 | 134.939 |
| misc | 46730 | 200.509 | 200.468 |

GPU duration sums overlap across ranks and cannot be subtracted from request wall time. AFTER shares of summed observed kernel duration: misc 45.87%, named GEMM/GEMV 30.88%, FP8 decode 23.25%. `gemm_gemv_named` is a name-based kernel category, NOT GemmEx API attribution. Pack count 0 means no matching kernel observed in this locally closed formal export, not guaranteed absence under incomplete collection.

Launch fragmentation: bf16_boundary 29,688; cuBLAS GEMV family 17,536; decode_weights 14,980; split_pair helper 7,364; remaining kernels 16,824. Thus large launch count is not all GEMM or remaining weight decode. Boundary conversions/helper kernels, many small math calls and 70,134 memsets remain. This profile does not prove a wall-critical-path optimization or authorize a production change.

## Historical public baseline, not a controlled allocator comparison

Historical cache-off public profile: 117,232 launches (14,654 per 8-output-token normalization), decode 26,592, pack 19,228. Historical cache-on public profile: 93,582 (11,697.75/token), decode 22,092, pack 78. Current BOTH 9ef and 7e1c: 86,392 (10,799/token), decode 14,980, pack 0 observed. Versus public cache-on: launches -7.683%, decode count -32.193%; this reduction already exists in BEFORE and is NOT attributed to V3 retirement. Normalization includes 5-token prefill and both ranks; it is not a pure decode-step count. Historical source/cache implementation, logging and server-marker windows differ; no historical wall-time causal comparison is made. Original report/raw pins were verified, never rewritten.

## Copies and interval gaps

Formal observed transfers are identical BEFORE/AFTER, per GPU: H2D 7,122 calls / 10,940,972 bytes; D2H 1,662 / 20,433,956 bytes; D2D 4,358 / 321,699,840 bytes. Both GPUs: 26,284 copies and 70,134 memsets. Per-direction device durations are recorded separately from host API duration.

Whole actual-HTTP window: BEFORE 2.729796217 s, AFTER 2.490041838 s, under profiling only. Union of all exported kernel/copy/set intervals across the two GPUs: 417.027 / 418.557 ms. Its complement is 2,312.769 / 2,071.485 ms (84.72% / 83.19%). This is precisely **no exported GPU activity**, NOT proven GPU idle, CPU execution, collective wait, or removable wall cost; collection can lose paired rows.

AFTER has two cuMemcpyDtoHAsync_v2 host calls of about 757 ms each on separate rank threads. Their summed duration is 1,514.784 ms but interval union only 757.469 ms. Each matches a successful 65,536-byte GPU D2H activity of 4,032 / 4,320 ns. Actual transfer duration cannot explain the long host span; cause remains unknown without scheduling/CPU stacks. Do not call this D2H bandwidth time or add the rank sums to wall time. BEFORE has no exported host CUDA API interval >=1 ms, despite a similarly large no-exported-activity gap. This further prevents a simplistic host-bound/cost attribution. Details: formal-host-api-outliers.json and launch_thread_exported_intervals.

## Independent allocator ledger and source review

AFTER each owner: ready batches/fences 1/3 → shutdown 50,447/151,341. These lifecycle counters include untraced hello/canary plus warmup and formal request; they are not the formal-window CUDA event counts. BEFORE lacks these fields: null/unavailable, never zero. All ready/shutdown snapshots: pending retirement 0, arena backing 3 GiB, no allocation/ordinary-pressure/quarantine failure. Pending/Unknown peaks remain null.

Fixed immutable artifacts remain 9ef9841598b4694fb611250bf8876c4314d252400136ddb23f606a979cadb656 and 7e1c23a88143cb6455e3c6e10bfe43a567fc97b5fd45745800eae3a1e39a0e73. Same TP2 thread, ctx128/batch8/seq2, workspace4096 MiB, cap22528 MiB, arena3 GiB/owner, info/rankperf off. Actual model-PID sampled environment verifies selected GPU UUID0/1. Current reviewed production delta is exactly allocator.rs/runtime.rs plus the fixed cfg(test) patch, not a generic source allowlist. Existing builder unset-to-0 oracle environment difference was already explicitly reviewed as disabled in both, not hidden. All pre-existing workflow pins still match. Source-only scope is not proof that timing has no other factors.

## Custody and execution outcome

Exactly one BEFORE and one AFTER profile, no retries, no new smoke/benchmark/production modification. Original B0: eight cards 0 MiB. Stable B1: selected cards 4 MiB each, six nonselected cards 1 MiB each; within unchanged 16 MiB cap. During model phase the existing 1 MiB nonselected-relative bound remained enforced. Both models returned to B1; collectors exited; authenticated supervisors exit0/not-timeout/ECHILD=true; original eight-card B0 returned. All eight instrumentation locks released, independently reacquired/verified/released at final check. Sampled GPU budgets passed; host-PID GPU attribution and exact continuous peaks remain unproven. No foreign process was killed.

## Cancel and CPU reporting records

First cancel: launch-tool phase, source unknown; subsequent inspection found no launched process/session/receipt. Renewed authorization launched BEFORE and AFTER exactly once. Second cancel: offline script-creation phase, script had not been created; both profile commands had already exited0. No subsequent GPU startup occurred. Both observations are separate JSON records; neither is labelled a user cancellation or CUDA failure.

CPU preflight: 67 tests passed. Offline interval math: 6 synthetic checks passed. One offline reporter KeyError exposed that profile-server.json only records a restricted environment subset; correction reads the actual model PID diagnostics in existing samples. Both environment checks passed; no runner/production or raw receipt modification, no model rerun.

## Files

- before-formal-analysis.json / after-formal-analysis.json: API returns/times, copies, kernel/name/launch-shape groups, per-UUID counts, explicitly scoped unions and gaps.
- formal-comparison.json: exact observed shape/text/token comparison and pinned historical public baselines.
- lifecycle-custody-allocator.json: B0/B1/B0, ECHILD, resource observations and independent allocator ready/shutdown ledgers.
- source-build-review.json: original artifact/build/source/runtime/smoke and fixed source delta pins.
- raw-file-pins.json: 130 original run files including .nsys-rep, SQLite, server logs, samples, HTTP/SSE, clock anchors, collector receipts and original audit/diagnostics.
- owner-closeout.json and seal-manifest.json: final evidence index. No full-trace or benchmark acceptance is issued.
