# Bounded checkpoint read pipeline

Implementation: `src/checkpoint/read_pipeline.rs`, exported from
`ferrule_model::checkpoint`. The tests in `mod.rs` are included as library unit
tests so fault injection stays private and absent from production builds.
`numeric.rs` is included by the numeric module for private-descriptor fault tests;
`main.rs` is also an automatically discovered public-API integration test target.

## Contract

- One OS worker executes the existing `VerifiedReadSession` positioned-read
  path. The request channel has depth one; at most one additional request waits
  behind the currently running read. Neither `poll` nor `take` reads a file.
- The worker always uses `Builder::stack_size`. `new` preserves its existing
  signature and supplies explicit default options (1 MiB requested, 1 MiB cap).
  `with_options(reader, limits, ReadPipelineOptions { worker_stack_bytes,
  max_worker_stack_bytes })` makes the requested stack and its independent cap
  caller-configurable. Zero, over-cap and host-overflow requests fail before
  spawning. Both scheduling modes must use identical options and quotas.
- `budget().worker_stack` reports `configured_bytes`, `limit_bytes`,
  `reserved_bytes`, `peak_bytes`, `unknown_bytes`, and `os_overhead_bytes`.
  Stack remains charged until a successful join; failed joins retain the
  reservation as unknown. A spawn failure proves no worker started and no read
  was admitted, so its startup stack reservation can be released. Snapshotting
  or cancelling a live read neither waits for I/O nor releases its stack/read
  reservation. `requested_host_reservation_bytes()` sums payload and requested
  stack charges as `u128`, without overflowing on virtual large-byte tests.
  **This is requested-size accounting, not an RSS or actual mapped-stack bound.**
  OS minimums/page rounding, guard pages, TLS, thread-runtime metadata and
  allocator overhead remain unobserved (`os_overhead_bytes == None`). Reserve
  an external platform allowance if a process-wide host cap must cover them.
- Admission reserves `plan.storage_bytes()` and one custody slot. This includes
  every extent of a paired numeric FP8 plan, not just its weights. The reader's
  batch limit and host address-space limit apply before dispatch/allocation.
- `ReadTicket` captures a process-wide monotonic `OperationId`, a nonzero caller
  source-image `SourceGeneration`, and the **entire original plan** (ordered
  extents and catalog snapshots). No source snapshot is recaptured by the worker.
  The generation is an identity label, not a freshness capability. Worker
  publication and owner consumption match the complete ticket.
- A source-verified result is packed in extent order and published only after
  the session's mandatory original-path **and pinned-handle** post boundary.
  This proves physical source freshness at that boundary, not numeric payload
  validity, continued freshness forever, or device installation.
- Success, ordinary failure and proved cancellation stay charged until the
  caller drops `ReadCompletion`. Taking/draining transfers custody; it does not
  refund it. Payload access is borrowed, so it cannot escape the guard as an
  independently owned vector. Downstream copies/decoded data need a separate
  budget. Plan metadata and allocator overhead are not payload-byte charges.
- Cancellation before worker start skips the read. Once started, the session
  finishes (including the post boundary), then successful bytes are suppressed.
  A genuine read/source failure is not hidden by a cancellation request.
- Panic, unexplained worker exit, or completion identity mismatch makes affected
  accepted work `Unknown`. It closes admission and retains its byte/slot charge.
  `take` refuses unknown custody and `drain` leaves it alone. `shutdown` joins the
  worker and reports unknown tickets/receipts without treating join or channel
  closure as a successful read proof. Unknown entries cannot be reused/refunded
  in that pipeline. A higher-level shared quota must carry these quarantined
  charges forward; creating another pipeline is not a release proof.
- `shutdown` stops admission and requests cancellation, but does not interrupt
  a positioned OS read. It can therefore block. Proved results remain takeable
  after shutdown. Dropping the pipeline joins it and discards untaken proved
  results; caller-held guards retain their payload and charge independently.
- There is no host payload cache, decoder continuation, inference engine, FP4
  provider, GPU scheduling change, or new dependency.

## Owner integration

1. Construct `CheckpointReadPipeline::new(reader, ReadPipelineLimits { ... })`
   using an existing bounded `CheckpointTensorReader` and the owner's host quota.
   Normal double-buffering needs at least two slots and enough bytes for both
   simultaneously held plans. Handle `QueueFull`/`BudgetExhausted` as backpressure.
2. Keep the existing semantic descriptor, and call
   `submit(source_image_generation, original_plan.clone())` before current
   compute. For numeric FP8, use the whole `NumericFp8Read::read_plan()`.
3. Run current compute on the existing owner/provider. Do not wait for next-read
   readiness before launching current compute in the overlapped schedule.
4. Use `poll(&ticket)` while independent work is runnable. For asynchronous
   notification, arm `listen()` **before** polling/draining, then await that owned
   listener only if no ready work was found. It uses the existing single-owner
   `CompletionHub` contract. `wait(&ticket)` is the explicit blocking alternative.
5. Inspect the typed `ReadPoll`: `Ready` is terminal but may contain a failure or
   cancellation. `Unknown` is quarantined custody, not a retry/refund signal.
   Call `take(&ticket)`, inspect `outcome()`, and keep the guard through consumption
   or an upload's real completion boundary. The pipeline does not prove DMA done.
6. Numeric FP8 conversion now has a completion-derived path:
   - Whole pair: `NumericFp8Artifact::from_verified_read(&paired, &completion,
     &original_ticket)` or `paired.materialize_verified_read(...)`.
   - Projection in a concatenated batch:
     `NumericFp8Artifact::from_verified_read_range(&paired, &completion,
     &original_ticket, packed_byte_range)` or
     `paired.materialize_verified_read_range(...)`.
   The original ticket includes operation ID, source-image generation, entire
   plan and snapshots; it cannot be publicly manufactured. The range is in
   packed **bytes**, must follow exact extent boundaries, and must match the
   selected descriptor's ordered physical extents and catalog snapshots.
   Concatenate all three complete paired plans (weights then scales for each)
   without coalescing extents, and use their checked prefix-sum byte ranges.
   The completion's mandatory pinned-handle worker post boundary is required;
   conversion checks the whole batch's original source snapshots both before
   and after the same numeric length/type/layout/value validation as normal
   reads. Changed sibling projection sources therefore invalidate conversion.
   No public arbitrary-slice constructor claims verified provenance.
   The helper borrows the completion and copies its selected weight/scale slices
   directly into the final immutable `Arc<[u8]>` payloads, with no intermediate
   slice-to-Vec-to-Arc buffers. It does **not** release or transfer read custody.
   Before calling, reserve staged artifact storage for the sum of selected
   `paired.read_plan().storage_bytes()` values while the full raw completion
   remains charged. Keep that external staged quota alive with the artifacts
   and through actual upload completion (including any Arc aliases). The read
   guard can be dropped once conversion/its other consumers finish; doing so
   does not free or budget the separately owned artifacts or prove DMA done.
7. Drop proved completions once their consumer is actually finished. On teardown,
   call `shutdown()`, record its unknown custody report, then `drain()` and drop
   proved results. Do not silently replace a failed worker to bypass its quota.

Blocking and overlapped comparisons use the **same** worker, plan, limits,
validation, payload and custody rules. The blocking schedule waits immediately
after `submit`; the overlapped schedule waits after current compute. Neither
makes the synchronous runtime forward return nonblocking.

## Timing receipts and evidence required

`ReadReceipt` exposes these exact fields:

- Identity/accounting: `operation`, `source_generation`, `requested_bytes`,
  `io_bytes`, `payload_bytes`, `cancellation_requested`.
- Host monotonic timeline: `submitted_at`, `worker_started_at`, `io_started_at`,
  `io_finished_at`, `source_verified_at`, `completed_at`, `taken_at`.

`io_bytes` is `Some(n)` only if all requested extents were successfully packed;
it is not a partial-read counter. `payload_bytes` is zero on failure/cancellation.
The I/O interval encloses allocation and actual positioned read calls, excluding
session opening and source checks. Optional timestamps are absent for phases
that did not produce a proved receipt; unknown work must not be presented as a
successful transfer. Instants are process-local, not serializable wall time.

Choose one host `Instant` trace origin before submitting work and export each
available timestamp as a duration from that origin. Also record current-compute
launch/completion host instants, **actual GPU event timestamps** for that compute,
and their host/device clock correlation. Pair events by operation ID, source
image generation, ticket plan/extent identity, and current compute identity.
`receipt.io_overlap(compute_host_start, compute_host_end)` gives the read-envelope
intersection, but a host enqueue/synchronization window alone does not prove GPU
execution overlap. Report queue delay, I/O time, post-verification time,
owner-visible readiness/stall, byte counts and peak/quarantined custody for both
schedules using identical work and budgets.

The deterministic CPU overlap test gates the real worker around its positioned
read calls: actual reader byte counters advance inside the owner window without
an owner poll or wait. Gates intentionally perturb timing and are not benchmark
results. These tests close the bounded/read-worker correctness claim only.
Real GPU/paper performance closure still requires consumer wiring and correlated
GPU evidence; neither is implemented or claimed here.

## Narrow CPU validation

```sh
cargo test -p ferrule-model --lib checkpoint:: --no-default-features
cargo test -p ferrule-model --test checkpoint_read_pipeline --no-default-features
cargo test -p ferrule-model --lib checkpoint::read_pipeline::tests --no-default-features
cargo test -p ferrule-model --lib checkpoint::numeric_fp8::completion_tests --no-default-features
cargo test -p ferrule-model --test checkpoint_read_session --no-default-features
cargo test -p ferrule-model --test checkpoint_numeric_fp8 --no-default-features
cargo check -p ferrule-model --no-default-features
cargo clippy -p ferrule-model --no-default-features --lib --tests -- -W clippy::all
rustfmt --edition 2024 --config skip_children=true --check crates/ferrule-model/src/checkpoint/read_pipeline.rs crates/ferrule-model/src/checkpoint/mod.rs crates/ferrule-model/src/checkpoint/numeric_fp8.rs crates/ferrule-model/tests/checkpoint_read_pipeline/mod.rs crates/ferrule-model/tests/checkpoint_read_pipeline/numeric.rs crates/ferrule-model/tests/checkpoint_read_pipeline/main.rs
```

Fixtures are 8–64 bytes. Gates use finite watchdogs; overflow tests use virtual
metadata and never allocate their large declared extents. No GPU is needed.

## Reentrant wake teardown and retained joining custody

`CompletionHub` invokes arbitrary user wakers synchronously. Checkpoint code does
not hold its state or joining mutex while notifying/closing that hub, including
mutex/condvar poison recovery. Poisoning quarantines queued/running entries,
unlocks first, then closes/wakes and reacquires as needed. Quarantine reasons
remain sticky when a late worker result arrives. User-waker panics are contained;
pending work is quarantined rather than permitting a double-panic process abort.

Capture `let join_owner = pipeline.join_handoff()` **before** registering a waker
that might drop/shut down the pipeline inline. Ordinary off-worker `shutdown`
and `Drop` still join synchronously. Worker-thread teardown instead closes the
queue, requests cancellation, and transfers the actual `JoinHandle` into retained
custody; it neither self-joins nor detaches. Its `ReadShutdown` explicitly has
`join_pending == true` and `join_handoff: Some(...)`. No worker-stack reservation
or untaken result is released by that self-drop. An owner must finish
`join_owner.join()` (or the reported handoff's `join()`) **off the read worker**.
This waits for actual worker exit, then permits stack release on successful join
and disposal of proved untaken results if their pipeline owner was destroyed.
Caller-held completions and unknown charges remain independently retained.
A join from the worker or while another owner is joining returns pending,
without blocking, consuming the handle, or fabricating a termination proof.
Check `join_pending` as well as `join_failed` and `unknown` before declaring
teardown complete or releasing a higher-level quota.

Discarding all handoffs after worker-side destruction is not a release proof:
the bounded handle/state/payload custody deliberately remains retained rather
than silently detaching or refunding. The supervising owner must preserve the
handoff, complete the join, and carry any remaining unknown charges. No hidden
reaper thread or additional OS stack is introduced. Requested-stack accounting
and its unobserved OS-overhead caveat above remain unchanged.

Reentrant drop/shutdown/poll/poison/unwind tests run in subprocesses with a
10-second external deadline. Thus a regression cannot hang the parent test suite
while unwinding or awaiting a self-join.

## Metadata identity v2 and immutable source-image requirement

Unix file snapshots now compare dev/inode, length, mtime **and ctime seconds and
nanoseconds**, for the original directory entry and pinned descriptor. This
rejects same-inode byte changes when size and mtime are restored, both before
reading and at the session/normal numeric/conversion post boundaries. A rename
away and back can also change ctime: it does not make an old snapshot current.
An existing stale session continues to pin its bounded FD but cannot publish
bytes as fresh; a newly captured catalog generation requires a separately
verified handle, without refreshing or evicting a pinned old-generation entry.
Related restored-path session tests retain those pin/cap assertions.

The metadata hash encoding is `ferrule-checkpoint-file-metadata-v2`; physical
source identity uses `ferrule-checkpoint-physical-source-v2`. ctime coordinates
participate in equality, ordering, Rust hashing and the versioned binary hash
encoding, hence newly built source generations. Fields remain private behind
`CheckpointSourceFileIdentity`; no public struct/wire deserializer or broad data
migration was added. Existing captured values and receipts are not rewritten.
The semantic descriptor/content-hash domain and SHA-256 primitive are unchanged;
new binaries must build new catalog/source pins rather than interpret old pins
as the stronger v2 metadata identity. An independent SHA-256 metadata vector is
pinned by the tests.

**This is bounded metadata identity, not a tensor-content hash, immutable-file
capability, or NFS coherence guarantee.** ctime depends on the filesystem's
metadata accuracy and resolution. Non-Unix retains the available length/mtime
and canonical-path fallback, without pretending to expose Unix change times.
Checkpoint images must remain immutable through read, validation, staging and
consumption; network/NFS storage must additionally provide coherent metadata
(or an external immutable-image/snapshot guarantee). Metadata-only checks do
not establish arbitrary adversarial or cached-network content equivalence.
