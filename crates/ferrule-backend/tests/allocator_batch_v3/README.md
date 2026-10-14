# Allocator batching V3 — CPU qualification, GPU not run

V3 fixes three independently reviewed defects in V2. V1/V2 freezes and receipts
remain historical, unchanged; they do not qualify these new sources. No agent
currently owns a GPU gate. This delivery executes no GPU test or benchmark.

## Production scope / API

Only `src/cuda/allocator.rs` and `src/cuda/runtime.rs` change. Model/runtime
crate, composition, decode cache, native/kernel sources and geometry, numeric
references, paper values, dependencies, and build scripts are unchanged.

No public API signature or metric field is added in V3. V2 teardown state and
metrics remain in force. The backend-private recording return type changes to
`Result<Option<Vec<CudaRetirementEvent>>, ()>`: `Some(events)` is a complete
recording, `None` is pending stream teardown, and `Err` uses the existing
error-only completion fallback. `CaptureState::completion_unknown()` permits
one capture-state guard to distinguish temporary capture from terminal Unknown.

## P1: stable stream teardown coverage without a strong-reference cycle

Every registration stores a Weak stream owner plus an independently owned atomic
completion ticket. The stream also holds that ticket. Initial state is Pending,
**before** the last stream Arc can disappear; therefore the interval before the
Drop body starts is covered too. The ticket has no context/stream pointer or
strong owner. There is no context -> allocator/registry -> stream -> context
strong cycle, no per-block stream vector, and no per-block event recording.

- A live Weak upgrade supplies temporary strong custody during the existing
  per-stream event recording, as before.
- Weak gone + Pending means the destructor has not proved synchronization.
  Recording returns `None`, preserving the unfenced batch. No event is created,
  no empty set is treated as complete, and no context sync is added.
- Weak gone + Complete may be omitted only after the stream's existing
  synchronization returned success. The Release store / Acquire load publishes
  that completion, not the mere fact that destruction began. Native stream
  destruction may still be finishing; prior work is already proven complete.
- Failed/unproven stream teardown becomes Unknown, with a tracing error. It is
  never treated as successful completion. Existing Unknown stream owner-retain
  and V2 final-context quarantine rules remain fail-closed.
- Registration and recording prune only Complete tickets. Pending/Unknown Weak
  entries are not discarded. Temporary strong-stream snapshots are dropped
  outside the registry mutex, so their destructors cannot run under that lock.

Tickets are per registered stream and cleaned on subsequent registration/poll;
there is no global runtime retirement framework. Final context shutdown still
uses its existing full-context completion proof. All raw-pointer submission
must still finish on registered streams before final block-owner Drop;
unregistered streams and late external use remain unsupported.

## P2: capture is a publication barrier, not loss of proof

Polling still installs Retained before driver calls for unwind safety. At the
publication check, only terminal Unknown discards the local result. Temporary
capture restores the existing Recorded or Complete state, keeps its fence/proof
accounting, and delays extent publication. After capture ends, the next poll
uses that original proof, without recording replacement events or requiring an
extra synchronization. Actual failed query + failed completion fallback remains
terminal Unknown as required by V2; success of the old fallback remains valid.

## P3: growth backpressure, never a replacement admission budget

Allocation first searches compatible safe free extents. Only the `None` branch,
which would call driver allocation, checks the existing 256-batch / 256-MiB
ordinary retirement high watermarks. Safe reuse remains allowed at either
watermark, including a previously admitted/reserved 22-GiB extent.

Capture quarantine is excluded from this ordinary-progress throttle: polling
cannot reclaim it, so charging it would create an unrecoverable retry loop.
It remains fully charged in capture/pending/reserved metrics and is not reused.
Graph-captured resources retain their existing final-context release policy.

Errors distinguish:

- active capture: retry after capture ends;
- terminal completion Unknown: quarantined resources, restart worker;
- ordinary retirement pressure with no compatible free extent: driver growth
  deferred; poll after stream progress, explicitly not a device-memory budget.

Batch buffering remains 64 blocks / 64 MiB (oversized blocks standalone).
These limits and growth watermarks are **not** an absolute host-metadata or GPU
reservation cap. Safe reuse and already-live owner drops can increase pending
metadata beyond a watermark; existing/external admission remains authoritative.
No arbitrary 256-MiB budget is imposed on a previously admitted working set.

## CPU race/fault qualification

`main.rs` compiles the real runtime and allocator. The shared dedicated FFI mock
in `../allocator_context_drop/driver.rs` never forwards tested lifecycle calls to
CUDA. The stream race uses an actual last `Arc<CudaStream>` Drop on another
thread and a barrier inside mock `cuStreamSynchronize`. It asserts Weak upgrade
has already failed while the sync is still blocked; this is not a simplified
mock stream-count decrement.

Ten new tests (`tests::v3_` filter):

1. Weak gone / stream sync pending: repeated polls preserve bytes and create no
   fences/context sync; after successful teardown only, an empty set is valid.
2. Registering another stream during that window cannot prune the old ticket or
   reuse its block; after completion record only the remaining registration.
3. Failed stream teardown remains terminal; later local success cannot reclaim.
4. Capture starts during query of Recorded fences: keep the original events.
5. Capture starts as the query completes: keep Complete proof until publication.
6. Unknown during query stays terminal, unlike temporary capture.
7. 256 batches / 64 KiB pending: small and admitted 22-GiB free extents still
   reuse without driver growth; only new dedicated growth is deferred.
8. 256 MiB ordinary pending: standard safe reuse still works; growth is deferred.
9. 256 MiB capture quarantine: active-capture error is distinct; after capture
   ends new growth works while quarantine remains fully accounted and retained.
10. Eight blocks / two registered streams still create/record only two fences;
    normal context teardown has no strong-custody cycle.

The 22-GiB test uses fake addresses only: no GPU or host payload is allocated.
Existing V2 teardown tests (14), batch tests (29; pressure assertions updated),
and CUDA library allocator CPU tests (7) remain regression checks. The existing
GPU tests and `cuda_allocator` integration are compiled only. Listing ignored
GPU definitions is not execution or GPU qualification.

## Reproduction / receipt

```sh
. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void FERRULE_NO_CUDA=1
export CARGO_BUILD_JOBS=2 FERRULE_CUDA_TEST_ORACLE=0
export CARGO_TARGET_DIR=target/validation/allocator-batch-v3/cargo-cuda124
cargo test --locked --offline -p ferrule-backend --features cuda \
  --lib --test allocator_batch_v3 --test allocator_context_drop \
  --test allocator_retirement --test cuda_allocator --no-run
cargo test --locked --offline -p ferrule-backend --features cuda \
  --test allocator_batch_v3 tests::v3_
cargo test --locked --offline -p ferrule-backend --features cuda \
  --test allocator_context_drop tests::context_drop_
cargo test --locked --offline -p ferrule-backend --features cuda \
  --test allocator_retirement
cargo test --locked --offline -p ferrule-backend --features cuda \
  --lib cuda::allocator::
cargo clippy --locked --offline -p ferrule-backend --features cuda \
  --lib --test allocator_batch_v3 --test allocator_context_drop \
  --test allocator_retirement --test cuda_allocator -- -D warnings
```

No-default-features batch tests and targeted rustfmt are checked separately.
`FREEZE_V3.sha256` pins the V3 sources and test inputs. `CPU_RECEIPT_V3.json`
records commands, artifacts and CPU/build results. A future GPU run requires
new explicit ownership/authorization and a fresh V3 GPU receipt; no prior GPU,
throughput, event-count, or numerical receipt can be inherited from V1/V2.
