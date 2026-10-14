# Allocator/context teardown closure — qualification V2

This is a **new qualification**, not an amendment to the V1 receipt in
`tests/allocator_retirement/FREEZE.sha256`. That file and its README remain
historical artifacts, unchanged. This document supersedes their statement that
primary-context teardown is unprotected and their failed-query-fallback recovery
rule. GPU qualification is pending, with **agent322 as sole GPU owner**.

## Verified root cause and minimal production scope

The prior runtime already guarded Unknown on entry to `CudaContext::drop`.
Allocator shutdown also rechecked capture/Unknown after context synchronization.
The actual bug was the lost outcome: shutdown returned `()`, so Drop still
called `cuDevicePrimaryCtxRelease` after shutdown had refused reclamation.

Only production `src/cuda/allocator.rs` and `src/cuda/runtime.rs` change. No
model/runtime crate, composition, decode cache, kernel, provider, dependency,
build script, input, paper value, or runtime-wide ownership framework changes.
The existing test-only GraphFault hooks gain a context-sync fault and release
attempt counter; their shared observer exists only under `cfg(test)`.

## Completion / ownership contract

- `shutdown_with_context` now returns `CudaResult<()>`. `Ok(())` requires the
  existing context-wide completion to succeed, capture/Unknown checks to pass,
  the pre-sync snapshot to drain, **all** nonzero live/pending/capture owners to
  be gone, and idle-segment trimming to succeed. Errors are no longer discarded
  by final context Drop. This does not introduce another synchronization.
- Concurrent retirement beyond that snapshot or a remaining live owner is an
  incomplete drain, not a fake completion proof. Ordinary explicit shutdown
  keeps its public void signature and exposes the outcome through metrics.
  A later full drain may succeed only if the context has not become Unknown.
- Failed event query still uses the old event-synchronize fallback. A successful
  fallback remains valid completion evidence. Failed fallback now marks the
  owning context Unknown, so later local event/context success cannot wash away
  failed completion. NOT_READY remains pending, not Unknown.
- Final Drop checks capture/Unknown before and after shutdown and propagates
  bind/free/release errors to its error path. Rust unwind during final teardown
  is caught; it does not turn into primary-release permission.
- On failure, Drop makes Unknown terminal and preserves an owning allocator
  `Arc`, including segment ledger, pending block metadata, and remaining raw
  retirement-event custody. It does not merely omit primary release and then
  discard the resource ledger. No recovery or allocator reuse is authorized.
- Context capture/graph and stream owner-retain rules remain unchanged. A
  primary release failure after valid quiescence is reported and not retried;
  the code does not claim the driver's failed release changed no internal state.

This intentionally quarantines resources until process exit. Recovery is worker
restart, not another fence or a silent in-process reset. No unproven GPU resource
is freed to satisfy a memory watermark.

## Typed state, metrics, and logging

Allocator-private `AllocatorShutdownState` has `Running`, `Quiescent`, and
`Retained` states. Before shutdown driver calls it is conservatively Retained,
including on unwind. Public `CudaAllocatorMetrics` adds two bool fields:

- `shutdown_quiescent`: shutdown established the complete drain/release proof.
- `shutdown_retained`: shutdown/final teardown did not establish that proof.

Both are false before the first shutdown attempt. Retained on an incomplete
ordinary drain is not by itself a new graph Unknown; driver completion Unknown
is sticky through the existing capture state. Adding fields can affect external
exhaustive metrics struct literals. No public method signature changed.

Final Drop emits `tracing::error!` on the existing tracing route, target
`ferrule_backend::cuda::runtime`, with ordinal/device, error, retained state,
reserved bytes, pending blocks/bytes, unfenced bytes, capture bytes, and an
explicit worker-restart/investigation action. It never polls for log metrics.
An already-Unknown context reports the existing capture/Unknown preflight
reason; a newly failed shutdown preserves the original CUDA operation/code.
Failures of record/context-sync fallback and query/event-sync fallback also log
at the allocator failure point, including retained block/byte counts and the
available original errors. This is necessary because the unchanged Unknown
stream owner-retain rule can keep Context Drop from ever running. No timing
probes or successful-path logging are added.

## CPU qualification

`main.rs` compiles the **actual runtime and allocator sources**, rather than a
copy of Drop. `driver.rs` exports CUDA FFI mocks only in this dedicated test
executable. All lifecycle calls exercised by the 14 selected tests are handled
there, with no forwarding to CUDA. Thread-local mock ownership permits concurrent
tests. The fixture is never linked into the production library or GPU gate.

Run only the exact CPU filter shown below; unrelated imported runtime tests are
not qualification evidence. `tests.rs` covers:

1. Quiescent sync -> free -> primary release, each exactly once.
2. Final sync failures 3/700/719/999: no free/release, no unwind, actionable log.
3. Unknown before Drop: no sync/release; recorded event ledger is retained too.
4. Failed shutdown becomes Unknown; later successful codes cannot wash it away.
5. Query failure + failed fallback becomes terminal Unknown; no later reuse.
6. Query failure + valid old fallback remains releasable.
7. Unknown arising during a successful drain sync defeats that local success.
8. Concurrent last-owner retirement requires another complete drain.
9. Active capture retains capture quarantine.
10. Context-binding failure retains unfenced storage and reports the operation.
11. Trim failure preserves reserved-byte accounting and prevents primary release.
12. Primary release failure is logged without retry.
13. Query/fallback failure remains observable when an Unknown stream retains the
    context owner and final Context Drop never runs.
14. Record/fallback failure logs its context-sync error under that same existing
    stream owner-retain rule.

The existing batch target remains 29 tests; its mock synchronization now mirrors
real runtime Unknown on failure, and obsolete later-success recovery assertions
were updated. Seven allocator CPU unit tests also run in the actual CUDA library.

Hidden-device commands, CUDA 12.4.99 / sm_86 / production oracle disabled:

```sh
. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void FERRULE_NO_CUDA=1
export CARGO_BUILD_JOBS=2 FERRULE_CUDA_TEST_ORACLE=0
export CARGO_TARGET_DIR=target/validation/allocator-context-drop-v2/cargo-cuda124
cargo test --locked --offline -p ferrule-backend --features cuda \
  --lib --test allocator_context_drop --test allocator_retirement \
  --test cuda_allocator --no-run
cargo test --locked --offline -p ferrule-backend --features cuda \
  --test allocator_context_drop tests::context_drop_
cargo test --locked --offline -p ferrule-backend --features cuda \
  --test allocator_retirement
cargo test --locked --offline -p ferrule-backend --features cuda \
  --lib cuda::allocator::
cargo clippy --locked --offline -p ferrule-backend --features cuda \
  --lib --test allocator_context_drop --test allocator_retirement \
  --test cuda_allocator -- -D warnings
```

No-default-features `allocator_retirement` and targeted rustfmt are checked
separately. The existing `cuda_allocator` integration test and the four GPU
teardown tests are **compiled only**, not executed by this agent.
See `GPU_GATE.md` for agent322's small gate, `FREEZE_V2.sha256` for the new source
freeze, and `CPU_RECEIPT_V2.json` for current CPU/build results. GPU receipts must
be new artifacts referencing this freeze, never overwrite V1 qualification.
