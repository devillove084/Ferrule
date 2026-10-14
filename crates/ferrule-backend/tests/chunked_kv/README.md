# HostKV physical backend: first usable path

## Opt-in and default

`CudaPagedKvPool::new` remains the fixed contiguous pool. Select
`CudaPagedKvPool::new_chunked(context, planes, page_size, max_slots,
staging_page_limit)` explicitly. The current transactional schema remains two
matching F32 `standard_gqa.key` / `standard_gqa.value` planes. No model weights
are copied to CPU and no logical page manager or placement registry is added.

Each occupied physical slot owns one **dedicated allocator segment per plane**.
Segments are not shared with weights or scratch, so freeing a small KV page does
not depend on freeing unrelated allocations. All existing slot mappings, slot
epochs, original/staged/COW pins and prepared retirement stay authoritative.

## Exact model hook

Continue `prepare -> enter -> active_view -> with_f32_gqa_planes -> leave` and
ordinary or prepared commit/abort. The callback exposes an Entry-owned compact
staging chunk, not authoritative page allocations. The callback's
`block_slots` / `block_offsets` are the **only kernel coordinates**. For
`PagedF32GqaLayout.physical_slots`, use `planes.kernel_slot_capacity()` rather
than pool capacity or `CudaKvView::physical_slot`. Native attention/scatter ABI
is unchanged and actually runs on these compact contiguous planes.

Every declared page (including COW sources omitted from the packed batch) counts
against `staging_page_limit`. Admission rejects over-bound or overlapping active
transactions before physical slot reservation. This initial path allows one
transaction per pool, bounding aggregate Entry staging to the configured bound.
The budget must include authoritative original + provisional pages **plus** the
staging copy, allocator grant rounding and host snapshots/pinned DMA frames.
This is not a constant-memory streaming-attention implementation: histories
larger than the declared bound are rejected, not silently materialized.

Gather is D2D after prepare's writes on the owner compute stream. `leave` copies
only staged writable pages back to their authoritative chunks, then records the
exact write/consumer fence. Err/unwind in a chunked callback is conservative
sticky Unknown; adapters must not publish a failed append. Old view epochs,
nonces and foreign pool identities still reject ABA. An escaped staging alias
retains only its staging grant, not the original/installed page allocation.

## Exact runtime hook

At an existing logical preemption boundary, with the runtime lease still held:

1. Finish ordinary/prepared transaction cleanup first. Call `pool.preempt(pages)`.
   Existing original/staged/COW, prefix and committed-reader pins reject release.
2. Backend pinned D2H copies **all tokens, all layers, both planes**. Exact owner
   stream completion is required before any logical mapping is detached. The
   backend retains its own full bit-preserving host snapshots.
3. `host_snapshot(page)` returns an owned copy for host consumers. No public
   completion ACK or caller-authored restore data is accepted.
4. Call `trim_physical_allocations()` later, after allocator retirements become
   ready. A zero result is valid; retry at a later maintenance boundary, not in a
   GPU-blocking spin loop. This is independent from logical slot release.
5. Before restoring logical residency, call `restore(pages)`. Dedicated chunks
   are allocated and full pinned H2D completes before mappings publish. Continue
   ordinary transactional attention using the same hook above.
6. Logical deletion uses `release(pages)`; deleting a preempted page drops its
   backend snapshot, not a nonexistent device allocation.

`physical_stats()` separates authoritative device payload, Entry staging payload,
backend host snapshots, retained pinned frames and context-wide allocator
metrics. `CudaKvReclaim` separates allocator trim return from the actual
successful `driver_free_bytes` counter delta. Neither is page-attributed;
context-wide trim may also free unrelated already-dead segments. Payload removal,
slot reuse, allocator retirement and driver free are **not interchangeable**.

On any uncertain producer/DMA/consumer result, retain logical runtime custody
when `has_unresolved_custody()` is true. `mark_transaction_unknown(tx)` is a
one-way cancellation/failure signal, never an ACK. Unknown cannot be cleared by
a later successful event. The pool intentionally retains its resources on final
Drop. This phase provides no recovery/reaper for Unknown.

## Current limitations / integration gates

- This change is backend-only; model/runtime/CLI opt-in is not enabled here.
- Graph capture is unsupported in chunked mode.
- Legacy `CudaCommittedRead` and append-prefix bridge use fixed plane addresses;
  chunked mode rejects them before taking new read custody. Use the ordinary
  transaction attention path, or separately adapt those bridges before enabling
  their dispatch in a chunked model.
- Snapshot/reload are synchronous owner-stream operations. Pinned frame usage is
  proportional to the explicit page batch; callers should use bounded batches.
- CPU attention can consume the owned snapshot after VRAM release; GPU attention
  requires explicit reload. No automatic placement policy is implied.

## Validation

All CUDA commands source
`target/validation/qwen35-35b-current/toolchain/env-cuda124.sh`, then override
`CUDA_VISIBLE_DEVICES=`. GPU fixtures are ignored and **not executed** here.

- `cargo test -p ferrule-backend --test chunked_kv --no-default-features`
  exercises production admission/compact mapping/sticky completion without CUDA.
- `cargo test -p ferrule-backend --features cuda --test chunked_kv --no-run`
  compiles real all-plane DMA -> release/trim -> CPU attention -> reload -> GPU
  attention vs fixed-pool numerical fixtures, COW/ABA/alias and cancellation cases.
- `cargo test -p ferrule-backend --features cuda --lib --no-run` also compiles
  private injected post-DMA/pre-fence failures and partial prepare failures.

Ignored GPU fixtures require a separately approved idle GPU run. Compilation
alone does not establish numerical correctness or measured VRAM savings.
