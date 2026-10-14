# V2 GPU small gate — agent322 only, not run here

**This file defines tests; it is not authorization for another agent to use a
GPU.** Agent322 is the sole owner. Use its assigned device mask, CUDA 12.4/sm86,
production oracle disabled, and a fresh qualification output directory. Logical
device 0 means the first device in that assigned mask, not an independently
selected physical GPU. The toolchain environment script sets a GPU mask, so
agent322 must restore its assigned mask after sourcing it.

First verify `FREEZE_V2.sha256` from the repository root. A mismatch requires a
new source freeze and qualification, not reuse of a prior receipt. Keep V1
receipts and frozen artifacts unchanged. CPU results do not qualify GPU behavior
or the original TP2 req6 throughput/event counts.

## Executable definitions

`gpu.rs` is included only by the real runtime's test module. **Use `--lib`, not
the CPU FFI-mock `--test allocator_context_drop` executable.** All four tests are
ignored by default, require a real CUDA device (no skip-on-missing-device), and
must run as separate exact-test processes. The allocator/context are real and
perform an actual device allocation, memset, and synchronization. Negative
status injection uses existing test-only GraphFault machinery; no invalid GPU
kernel or device-damaging fault is launched.

For each row, run:

```text
cargo test --locked --offline -p ferrule-backend --features cuda --lib   cuda::runtime::context_drop_gpu_tests::<EXACT_NAME> -- --ignored --exact --nocapture
```

Use a per-process timeout (600 seconds maximum). Do not combine the intentional
quarantine cases into a long-lived worker: their retained primary contexts and
GPU segments are reclaimed only when that test process exits.

| Exact name | Required evidence |
| --- | --- |
| `gpu_context_drop_quiescent_releases_primary_once` | shutdown_quiescent=true; pending/reserved=0; one driver free and one primary-release attempt |
| `gpu_context_drop_shutdown_sync_fault_retains_primary_and_segments` | real context sync then injected 999; shutdown_retained=true; pending block/bytes=1/256; reserved remains; zero frees/releases |
| `gpu_context_drop_unknown_before_does_not_attempt_release` | sticky Unknown on entry; zero shutdown-sync and primary-release attempts; quarantine unchanged |
| `gpu_context_drop_query_fault_cannot_be_washed_by_later_event` | real recorded/completed event, injected query failure; next query remains blocked; final Drop preserves allocator bytes and skips primary release |

The query GPU case exercises the public event Unknown path. The distinct
retirement-event query + failed completion fallback is covered directly by the
CPU real-runtime FFI test, not claimed as the same GPU test. Likewise, the
bounded CPU barriers qualify Unknown-during-drain and concurrent last-owner
retirement deterministically; the four-test GPU small gate does not claim an
exhaustive asynchronous-driver race or genuine hardware-fault campaign.

## Fresh receipt and stop conditions

Record source-freeze hash, compiler/target, production/oracle flag, assigned
GPU identity, each exact command, subprocess exit/timeout, stdout/stderr, and
asserted release/free/retained metrics in a new **V2 GPU receipt**. These are
return-status fault tests, not evidence that every real CUDA failure is safe.
A zero-test run, skipped test, stale source hash, timeout, primary release in a
quarantine case, or changed reference/kernel geometry is a gate failure.

Stop on the first failure and preserve its artifacts. No numerical tolerances,
model inputs, composition, paper numbers, or old performance receipts may be
changed to pass this gate. A pass only closes this teardown small gate; any
subsequent official TP2 req6 performance qualification needs its own authorized,
fresh run and cannot inherit prior event counts or tok/s claims.
