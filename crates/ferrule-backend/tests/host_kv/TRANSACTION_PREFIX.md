> **Latest implementation:** [RUNTIME_APPEND_BRIDGE.md](RUNTIME_APPEND_BRIDGE.md).
> The explicit borrowed runtime/model append hook is now connected; automatic
> runner dispatch and page eviction remain disabled. The internal-only slice
> and its earlier validation recorded below are historical, not current limits.

# Append-prefix DMA and consumer custody — production dispatch CLOSED

## Implemented

- `KvPageManager::derive_append_prefix` derives a non-cloneable borrowed capability
  from an existing live `KvReservation`. It does not call the quiescent committed
  lease API or register another logical pin. The manager and reservation cannot
  be mutated/consumed while that borrow is live. Reservation tokens now carry
  their manager identity, so equal manager-local IDs/views are not interchangeable.
- The common `HostKvAppendReadWitness` describes one appended query, a nonempty
  full committed prefix, sequence incarnation/generations, reservation identity,
  and the exact COW-projected block table. It is metadata, NOT physical authority.
  The caller-supplied execution transaction ID is a trusted composition input;
  the runtime manager does not itself own that model transaction ID.
- Crate-internal `CudaPagedKvPool::derive_append_prefix` derives an opaque,
  non-cloneable `CudaAppendPrefix` from the actual entered physical transaction.
  It authenticates pool/transaction nonce, pool-to-manager binding, source schema,
  layer, allocation identity, packed batch and physical slot epochs.
- `Entry::prefix_read` retains the subordinate read record and producer/DMA/
  consumer fence fields. There is no second page manager or transaction journal.
  The first accepted logical witness is fixed for the remainder of that Entry;
  subsequent reads may change layer but not the reservation/frontier/incarnation.
- Committed-prefix bytes come from immutable **original** slots. Shadow/COW
  destinations must copy exactly the declared source. All write slots are checked
  for disjointness from prefix reads, and both source and projected epochs are
  frozen. Existing whole-allocation mutable plane callbacks are blocked across
  the pool while a subordinate reader exists.
- Normal/prepared commit, abort completion, retirement/finish and re-entry cannot
  consume unresolved read custody. Leaving the owning transaction also requires
  the read to finish first. Handle drop is not an ACK. Missing submission fences
  and event-query failures produce sticky Unknown; pool availability/free capacity
  and unresolved-custody reporting reflect that quarantine.
- `merge_append_partials` validates the exact disjoint prefix/tail ranges and
  uses the existing unnormalized `(max, denominator, numerator)` partial merge.
  It rejects query-position, geometry and mask disagreement. It returns another
  partial; normalization happens only after the full merge. The trusted adapter
  must still certify that both operands use the same actual Q and declared ranges.

## Transaction-owned DMA / CPU consumer (current slice)

`CudaAppendPrefix` now has internal real-DMA hooks, not a metadata-only submit:

```rust
start_cpu_prefix(
    &mut self,
    operators: &CudaOperators,
    query: CudaF32Buffer,
    config: StreamingF32GqaConfig,
    budget: &PlacementBudget,
) -> Result<()>;
poll_cpu_prefix(&mut self) -> Result<bool>;
consume_cpu_prefix(
    &mut self,
    consumer: impl FnOnce(&StreamingF32GqaPartial) -> Result<()>,
) -> Result<()>;
finish(&mut self) -> Result<bool>;
```

- `Entry::prefix_dma` owns Q, the producer event, bounded transport, the exact
  outstanding ticket, CPU Q/K chunk/unnormalized partial and budget reservation.
  The external capability owns none of those resources. Drop is not an ACK;
  abandoning the entire pool retains its existing Entry/storage quarantine.
- Admission reuses the shared six-domain `PlacementBudget` and conservative
  `committed_gqa_credits`; unused GPU staging grants are deliberately overcharged.
  Existing resident allocation and caller Q remain caller-accounted, as in the
  committed path. No new manager, pin journal or budget pool is introduced.
- Both submission domains are registered before the first transfer. Poll calls
  real `submit_d2h_f32_after` on Q and original-slot K/V; the predecessor is the
  same operator owner's compute event. Q aliasing KV allocations is rejected.
- Successful `CudaTransferCompletion::D2H` must match the exact ticket identity
  (transport, direction, slot, generation). The transport queries that ticket's
  immutable recorded event. A canceled transfer is never successful data.
- The production `AppendStream` consumes each checked payload synchronously,
  bounds K/V chunks by page/block/frontier, and accumulates unnormalized partials
  at the appended query position. It never averages normalized prefix/tail rows.
- Ready data is not consumer completion. Only a successful synchronous partial
  callback produces CPU-consumed evidence. No fresh compute event substitutes
  for DMA or CPU completion. The caller must charge any retained output itself.
- Producer fencing plus exact DMA and CPU evidence precede explicit finish;
  resources are dropped before the shared charge is refunded. Early finish
  returns pending, not success. Transport/consumer failures and unwinding retain
  sticky Unknown, block pool availability and report zero usable free capacity.
  There is still no recovery reaper for this capability.

## GPU-owner / runtime handoff (dispatch still CLOSED)

The hooks remain crate-internal deliberately. The cross-crate bridge must retain
`ReservedAppendPrefix` from the actual live `KvReservation`, authenticate its
`witness().transaction()` against the same existing physical transaction handle,
and call `derive_append_prefix(&transaction, witness(), layer())`. Public witness
metadata alone is NOT logical authority. Do not expose a witness-only production
entrypoint or redirect append through the quiescent committed-read lease/provider.

The adapter must also retain the actual immutable post-RoPE Q source and certify
that both prefix and tail use it, with identical scale/mask/geometry. Passing an
operator owner authenticates the stream owner, not Q's semantic origin; an alias
held elsewhere must not be overwritten while Q D2H is pending. Existing whole-
allocation KV write callbacks remain blocked until prefix finish. Therefore
schedule staged append/tail work before deriving the capability (or use a future
narrow, proven-disjoint write interface); do not relax that guard for convenience.

After polling ready, synchronously merge the exact `0..committed_tokens` prefix
and `committed_tokens..committed_tokens+1` tail into separately budgeted output,
using `merge_append_partials`. Normalize only the combined partial. Async output
DMA/GPU consumers require an additional retained output/consumer fence path;
`consume_cpu_prefix` is NOT an async launch/ACK interface. Retain logical and
physical custody on errors, including a failed submit that returned no ticket.
No standard/cuda.rs, root, hybrid, numeric or native transport file was changed.

## Slot reuse versus actual VRAM release

The existing `CudaPagedKvPool::preempt` / `CudaKvPagePool::preempt` can download
all planes/layers, install an authoritative host snapshot under the existing page
ID, and return the resident slot to that same pool. It requires unpinned pages;
active transaction originals cannot be preempted through that path. Logical IDs
can survive this operation, with `page_status == Preempted`, until restore/release.

But `CudaKvPagePool::new` allocates one fixed-capacity device buffer per plane.
Recycling a slot **does not shrink/free that allocation**, so it permits capacity
reuse, not an allocator/VRAM-byte saving. This DMA slice does neither: it retains
only transient layer chunks/partial results, NOT a durable all-layer page image.
Resident counts and mappings are intentionally unchanged.

Before slot reclamation can join ordinary append, the same transaction/snapshot
store needs complete all-plane host custody, durable six-domain charges, proof of
all DMA/consumer quiescence, and a host-aware read/restore route that does not
require every protected logical page to be resident. Only then may the existing
mapping/epoch/free-slot authority detach a cold original while keeping active
shadow/COW writes separate. Returning actual VRAM additionally needs releasable
allocation granularity or another pool-allocation strategy, not just copying or
removing the resident mapping. No VRAM saving is claimed here.

First prefill, packed/multi-query batches, ordinary decoder selection, GPU/output
partial DMA and cancellation recovery remain closed.

## Current validation (no GPU execution)

- CPU `host_kv_protocol`: 19 passed, including the production chunk consumer,
  exact/duplicate/foreign ticket identity, malformed payload, consumer error and
  unwind, sliding-window prefix/tail merge, plus existing COW/epoch/fence tests.
- Focused Clippy with `-D clippy::too_many_arguments`: passed.
- `cargo fmt --all -- --check`: passed after the implementation and handoff edits.
- Standard `cargo check --locked -p ferrule-backend --features cuda --lib`
  with GPUs hidden and `FERRULE_CUDA_ARCH=sm_80`: blocked in build.rs because
  `nvcc` is absent. Native CUDA compilation/linkage is NOT validated.
- Temporary ignored `target/host-kv-prefix-rust-check/Cargo.toml` points directly
  at the production backend lib, disables its native build, and checks `cuda`
  Rust code only. `cargo check --offline --manifest-path ... --features cuda
  --lib` and `--tests` passed with `FERRULE_BACKEND_CUDA_COMPILED_TARGET=sm_80`.
  It emitted expected unknown-cfg warnings because it bypasses build.rs. This is
  not a replacement for the standard CUDA build or a GPU run.
- `tests/host_kv/append_dma_cuda.rs` is an ignored, type-checked physical fixture
  for the GPU owner: both shadow and COW, poisoned projected payloads to detect
  wrong-source reads, bounded real transfers, budget denial, pending consumer,
  write/preempt guards, explicit refund and unchanged resident capacity. NOT RUN.
- Runtime `--lib append_contract -- --test-threads=1`: both the 120-second
  attempt and subsequent 180-second attempt timed out during compilation;
  no test result is claimed for these runs. The earlier
  prerequisite's results below are historical, not this change's validation.

## Earlier prerequisite validation (historical)

`CUDA_VISIBLE_DEVICES='' CARGO_BUILD_JOBS=2 cargo test --locked -p ferrule-backend --test host_kv_protocol`

The existing target imports the production planner/read-record module and exercises
shadow/COW isolation, original/projected epoch ABA, foreign/stale tokens,
no-submit/drop semantics, pending producer/DMA/consumer fences, sticky Unknown,
source incarnation, strict prefix/tail ranges and partial merge. It does not use
an alternative manager or execute CUDA.

Runtime regressions are in `ferrule-runtime/tests/unit/cache/kv_read_authority/append_contract.rs`.
The cleanup rerun used a 600-second tool timeout:

`CUDA_VISIBLE_DEVICES='' CARGO_BUILD_JOBS=2 cargo test --locked -p ferrule-runtime --lib append_contract -- --test-threads=1`

All five tests passed after a 4m 02s build, including reservation/COW, foreign-owner,
and ABA coverage. Subsequent CPU validation passed 26 tests selected by `--lib
host_kv` (including those five) and four selected by `--lib committed_read_pins`.
The backend protocol and existing planner targets passed 15 and five tests.

`cargo clippy --locked -p ferrule-backend --test host_kv_protocol -- -D clippy::too_many_arguments`
passed with GPUs hidden. `CommittedChunk` reuses `KvPlaneDescriptor` for the checked
F32 range helper; `PrefixReadBinding` groups witness/layer/transaction incarnation
for derivation and validation. Neither helper needs a too-many-arguments allowance.
The geometry regression also covers zero page size, row overflow and non-F32 input.

The earlier prerequisite's backend CUDA compile-only check passed in the separate
`target/host-kv-cuda-check` directory. This cleanup deliberately ran no CUDA build
or GPU execution while the separate backend owner performs GPU validation.
