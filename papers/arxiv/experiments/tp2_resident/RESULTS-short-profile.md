# One short profiling attempt — blocked before model readiness

Run: `tp2-short-profile-20261009-001`; launch ID
`f3b126f0642a47658e84b17fc2b5baa9`.
Same Ferrule release: `e017a3a8c8ec4019a7f291a818643d9693e32b2dfee0a9cc503fcf84926c1b0b`.
**One attempt, no retry, no benchmark, no numerical full-capture or production change.**

## Tool capability and observed blocker

Nsight Systems 2026.1.3 is installed. `nsys status -e` reports CPU profiling
permissions available; `nvprof`, `perf`, `ncu` are absent from PATH. A hidden-GPU
`/bin/sleep` probe verified named-session `--start-later` → start → stop → report
export (not CUDA coverage). The GPU attempt reused the existing sealed
supervisor, sorted GPU0/1 locks, current observer, HTTP and graceful wrapper.
Requested traces were CUDA/NVTX/OSRT, all CUDA APIs; CPU sampling and context
switch sampling off, no system-wide GPU metrics. Planned capture was only one
warm QA8 plus one measured diagnostic QA8 after Hello/canaries/readiness.

At 3.179 s, observer rejected nonselected memory deltas against a zero baseline:
GPU2 **+4 MiB**, GPU3 **+4 MiB**, GPU4 **+2 MiB**, exceeding the unchanged **1 MiB**
tolerance. GPU5–7 each +1 MiB; selected GPU0/1 each +4 MiB. No compute PID or
utilization was observed in that sample. The root was the Nsight process and
its maps included driver/config/NVML libraries. Nsight discovery/driver overhead
is a plausible explanation, **not proven attribution**; foreign interference is
not proven absent by PID mapping. No tolerance relaxation or transport override.

`server.log` is empty, no `profile-server.json` was produced, no Ready, no HTTP
requests and no target CUDA report. The generic `server_starts=1` receipt counts
the **profiler root**, not proof of a loaded Ferrule server. No production CUDA
error was emitted. Synchronization counts/durations, kernel times, D2H/H2D bytes
and host/GPU gaps are **unavailable, not zero**. CPU probe report is not a model
profile. This is a concrete tool/custody-admission blocker, not a model regression.

## Three source-supported candidates, NOT measured bottleneck ranking

1. **Repeated completion fences.** `cuda/residency.rs:268–298` records and
   synchronizes a compute event, then synchronizes the upload stream.
   `cuda.rs:416–441` additionally drains boundary custody and syncs compute/upload
   streams. Mixed operations call this synchronous wrapper. Minimal candidate:
   coalesce provably redundant completion proofs only within the same owner and
   submitted-work generation, preserving failure poisoning, quarantine, retirement
   and publication. **Sync wait includes queued GPU work**, not pure host overhead.
2. **Host boundary round trips / rendezvous.** `hybrid_rank/mixed_layer.rs:800–836`
   reads BF16 rows to host, exchanges CPU words, then uploads BF16 output; output
   head gather does the same. `mixed_route_collective.rs:166–340` has exact-route
   agreement and `Condvar::wait_timeout`. Minimal candidate: reuse bounded pinned
   staging and coalesce compatible transfers while retaining BF16 rounding and
   fixed reduction/route agreement order. A device collective replacement is a
   separate arithmetic/lifecycle requalification, not an assumed safe shortcut.
3. **Decode/pack and route scratch work.** `bf16_gemmex.cuh:133–154` enqueues
   activation pack, FP8 weight decode, then GemmEx; pair path decodes both weights
   and splits output. `hybrid_rank/mixed.rs:655–728` allocates/zeros scratch and
   uploads route IDs/weights. Minimal candidate: preallocate bounded per-owner
   route buffers and remove provably unnecessary repacks; consider only a bounded
   decoded-weight reuse policy. Do **not** expand all FP8 weights to resident BF16
   under the existing 22 GiB/card budget or change expert accumulation order.

Existing `ferrule_performance=debug` invocation counters are implemented for
`CudaHybridModule` (`cuda/invocation_trace.rs`); this TP2 path uses
`MixedGqaInvocation`/`CudaStandardDecoderOperators`. No evidenced TP2 hook or
existing INFO counter receipt can supply the missing requested numbers. A future
minimal production instrumentation task should wire opt-in per-rank semantic
sync/copy/allocation counters into this path, not enable full numerical capture.
Native kernel time still requires a functioning profiler or explicit CUDA-event
instrumentation; facade kernel counts are not native kernel counts.

## Historical startup segments (not this failed profile)

Prior unprofiled F1/F2 used this same release, startup **385.979 / 367.234 s**.
Timestamp intervals from existing logs are retained in `source-startup-audit.json`:

- F1: resident model build entering owner-local admission → Mixed experimental source-bound admission: 72.168 s; Mixed experimental source-bound admission → Mixed owner preparation claim completion owner0: 128.825 s; Mixed owner preparation claim completion owner0 → Mixed owner preparation claim completion owner1: 101.732 s; Mixed owner preparation claim completion owner1 → Ferrule OpenAI server listening: 0.160 s.

- F2: resident model build entering owner-local admission → Mixed experimental source-bound admission: 70.843 s; Mixed experimental source-bound admission → Mixed owner preparation claim completion owner0: 128.882 s; Mixed owner preparation claim completion owner0 → Mixed owner preparation claim completion owner1: 100.599 s; Mixed owner preparation claim completion owner1 → Ferrule OpenAI server listening: 0.150 s.

These are coarse logged intervals, not exclusive upload/kernel/CPU attribution.
Startup outside these logged boundaries remains unclassified.

## Drain, preservation, next gate

The worker failed (returncode1), while authenticated descendant custody reached
`proven_drained`, no timeout, no cleanup errors. Pre/post idle passed. Final
independent two-UUID lock reacquisition and release verified **all eight GPUs
0 MiB, 0%, no compute apps**; no foreign signals. Tests: **83 TP2 +42 original PASS**.
Old baseline/smoke/performance receipts and their archived executed sources stay
unchanged; current generic adapter gained only the profiling hook. E2 remains
incomplete, `fairmatch=false`, `exact_speedup=null`.

- [Raw execution](runs/tp2-short-profile-20261009-001/execution.json)
- [Case and observer errors](runs/tp2-short-profile-20261009-001/runs/P1/session/case.json)
- [Raw samples/deltas](runs/tp2-short-profile-20261009-001/runs/P1/session/samples.jsonl)
- [Tool capability receipts](runs/tp2-short-profile-20261009-001/tools/capabilities.json)
- [Failure receipt](runs/tp2-short-profile-20261009-001/profiling-failure-report.json)
- [Append-only corrected nonselected delta projection](runs/tp2-short-profile-20261009-001/profiling-failure-analysis.json)
- [Source pins and prior startup intervals](runs/tp2-short-profile-20261009-001/source-startup-audit.json)
- [Final release](runs/tp2-short-profile-20261009-001/final-owner-release.json)

Before another GPU attempt: independently establish an Nsight target-only setup
that respects nonselected-device custody, or delegate the minimal TP2 counter
hook. Do not rebaseline away unexplained profiler allocations. Any fresh attempt
needs renewed authorization; this attempt is consumed.
