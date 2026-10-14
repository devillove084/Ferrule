# Bounded allocator retirement batches (GPU gate not run)

## Scope and API

Production change: `src/cuda/allocator.rs` only. No runtime, operator owner,
model, decode cache, composition, kernel geometry/source, native build, or
paper/reference changes. No dependencies, timing probes, last-use watermarks,
new escape tracking, or owner-provided completion proofs.

`CudaDeviceBlock` still retires only on its final `Arc` owner. DeviceBuffer
slices share their allocation owner in the unchanged runtime. Dedicated arena
owner aliases therefore contribute one retirement, not one per view. Raw
pointer users must submit all use on registered streams before dropping the
last owner. Late submission and unregistered-stream use remain unsupported.

No new methods or changed signatures. `CudaAllocatorMetrics` adds:

- `retired_blocks`: cumulative nonzero final-owner retirements, including capture.
- `retirement_batches`: cumulative detached ordinary batches, including failures.
- `retirement_created_fences`: fences returned by successful complete recordings.
  Runtime destroys partial recordings internally and returns only `Err(())`,
  so this is **not** an exact driver-create count on failure paths.
- `pending_retirement_batches`: detached batches, including in-flight/failed ones.
- `pending_unfenced_bytes`: open queue plus detached batches without a usable
  complete fence set (conservatively includes operations currently in flight).
  Capture quarantine remains separately reported by `capture_retirement_bytes`.

Existing pending blocks/bytes include open, in-flight, fenced, failed, and
capture retirements. Reserved bytes stay charged until actual driver free.
Adding public metrics fields can affect downstream exhaustive struct literals.

## Ordering, bounds, and behavior changes

Last-owner Drop enqueues a block without releasing its segment extent. Poll
(including existing allocation, allocator metrics, and trim callers) detaches
one open batch under the allocator state mutex, then records once per currently
registered stream through the unchanged runtime. All blocks in a batch retired
before the first record; later retirements go to a different batch. A later
fence covers at least the previously submitted work covered by immediate
recording. All fences must complete before any extent in that batch is released.

Each open batch is capped at 64 blocks / 64 MiB granted bytes. A block larger
than 64 MiB gets its own batch; blocks are not split. Count/byte pressure also
polls. At 256 detached pending batches or 256 MiB pending bytes (including
capture quarantine), allocation fails with a retirement-pressure error after
polling, rather than growing reserves or introducing a synchronization.
A caller may retry after registered-stream progress; this is a new allocation
failure condition, not a new automatic retry policy.

These are batch limits and admission high watermarks, **not an absolute cap on
all host metadata or GPU reservation**. Already-live allocations can retire
concurrently beyond a watermark, and existing free segment slack is still
cached by the original allocator. Their storage must remain retained. An
absolute cap under stalled/failed GPU progress is incompatible with preserving
all existing live owners, no new synchronization, and no premature release.
Allocation admission stops subsequent growth; existing trim reclaims idle
segments. No unlimited per-drop event multiplication is introduced.

Batch metadata stays reachable from allocator state across record/query/sync.
A per-batch try-lock prevents duplicate processing and lets concurrent or
reentrant polls skip in-flight driver operations. Event driver calls and event
Drop run without the allocator state mutex. Before driver calls, the batch is
marked retained, so unwind cannot turn it into an empty/complete event set.
An Unknown context blocks polling/reclamation. Free-extent publication holds
the capture-state guard; a concurrent capture or Unknown transition observed
after a driver call conservatively retains the batch. Partial recording or query
errors retain the entire batch unless the **existing** context/event completion
fallback succeeds. Failed fallback is terminal for ordinary polling; only a
later valid context-wide shutdown completion can release it.

Shutdown seals and snapshots before the existing context synchronization, and
releases only that snapshot. Concurrent later drops remain pending. Context
Drop already invokes this path; no extra per-batch context sync was added.
Stream Drop retains its existing synchronization/Unknown contract. Successfully
destroyed streams need no later event; Unknown blocks reclamation. Capture
retirements remain isolated until final context teardown. The allocator does
not add an independent destructor that would free unfenced segments.

Unchanged runtime teardown limitation: `CudaContext::drop` checks Unknown on
entry, calls allocator shutdown, then still calls `cuDevicePrimaryCtxRelease`.
It does not recheck Unknown newly raised by that shutdown's synchronization.
This allocator-only patch guarantees no allocator `cuMemFree` on failed
completion, **not** that the primary-context retain is preserved in that runtime
error path. The mock verifies allocator frees, not primary-context release.
Strengthening that separate contract needs an authorized runtime change; no
claim that all driver/context teardown failures are safe is made here.

`sync_stream()` itself is unchanged and does not poll the allocator. A compute
sync followed by the first poll may create fences that are not yet query-ready,
and it says nothing about unfinished upload/control work. Zero pending requires
all batch fences to complete and another poll (or valid shutdown), not a fake
already-complete proof. Immediate pointer reuse after a lone stream sync is not
promised by this change; existing CUDA reuse tests must be run in the GPU gate.

## CPU tests

`main.rs` compiles the **production allocator source** against `runtime.rs`, a
CPU-only mock. It neither loads nor invokes CUDA. Twenty-two batch tests plus
seven existing allocator tests pass (29 total):

- N=8 blocks, S=2 streams: exactly S creates/records; no pre-fence reuse;
  both streams required; shared event destruction and clean pending zero.
- Partial second-stream record failure retains all blocks/bytes; successful
  old record/query fallbacks still release; failed query/sync stays retained.
- Record/query/context-sync panic retains blocks and accounting.
- Unknown before polling, during recording/shutdown, and after failed completion cannot
  be overridden by local event success.
- Concurrent retirement during recording is a separate batch; reentrant
  poll/metrics do not deadlock on the allocator state mutex.
- Count, byte, oversized-block, pending-count, and pending-byte pressure;
  no new normal-path event/context synchronization; live-owner overshoot retained.
- Dedicated owner aliases retire once; zero-byte owners never retire.
- Shutdown only releases its pre-sync snapshot; context teardown flushes open
  queue; failed/panicking shutdown retains resources; capture stays isolated.
- Successful/Unknown stream teardown models preserve the old registry contract.
  These mocks do not validate CUDA stream destruction itself or real view DMA.

Commands (all with GPUs hidden):

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void FERRULE_NO_CUDA=1 \
  cargo test --locked --offline -p ferrule-backend --no-default-features \
  --test allocator_retirement

. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void FERRULE_NO_CUDA=1
export CARGO_BUILD_JOBS=2 FERRULE_CUDA_TEST_ORACLE=0
cargo clippy --locked --offline -p ferrule-backend --features cuda \
  --lib --test allocator_retirement --test cuda_allocator -- -D warnings
cargo test --locked --offline -p ferrule-backend --features cuda \
  --lib --test allocator_retirement --test cuda_allocator --no-run
cargo test --locked --offline -p ferrule-backend --features cuda \
  --test allocator_retirement
cargo test --locked --offline -p ferrule-backend --features cuda \
  --lib cuda::allocator::
```

## Validation result / freeze

CUDA compiler: 12.4.99, `sm_86`, production oracle disabled. The CPU-only target
passes both without CUDA features and linked in the CUDA feature build (29/29).
The production CUDA library's allocator-only CPU unit filter passes (7/7).
The listed targeted CUDA clippy (`-D warnings`), no-run compilation, and targeted
rustfmt check pass. The real `cuda_allocator` integration test was compiled,
not executed. All runs used hidden GPUs. 380 non-target backend/model/native
source hashes match the pre-edit snapshot. `FREEZE.sha256` pins this delivery's
allocator, tests, and documentation; paths are relative to the repository root.

## Pending authorized GPU gate

No GPU execution or throughput claim in this change. After authorization, run
real allocator/view/capture/teardown tests, then the unchanged official TP2 req6
workload. Compare driver create/record/query/destroy counts separately from
retired blocks/batches/fences, pending/unfenced/reserved peaks, and end-of-run
quiescence. Cross-thread summed event time is not wall time. Check admission
pressure does not cause workload allocation failure and check CPU contention.
Keep the model, kernel source/geometry, inputs, and numerical references frozen.
No update to paper numbers or prior tokens/s measurements is justified yet.
