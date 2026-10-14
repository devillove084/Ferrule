# Backend physical page placement boundary

## Status: contract + opt-in allocation only, not VRAM-saving execution

`src/physical_placement.rs` is a driver-independent subordinate custody contract.
The canonical public module is `ferrule_backend::physical_placement`;
`ferrule_backend::host_kv_placement` remains an alias of that same module.
`cuda::operators::kv::single_page_placement::CudaSinglePagePlacementAllocator`
explicitly allocates an independent F32 page/plane using the existing
`CudaF32Buffer -> DeviceBuffer -> CudaDeviceBlock` allocation path. It does not
allocate a private CUDA segment, bypass the canonical allocator, install a page
in a pool, export a kernel pointer, or launch page attention.

The existing transaction `Entry` and default `CudaKvPagePool` are unchanged.
**The new custody record is not yet embedded in the production Entry.** The
ordinary append path still owns fixed plane allocations and releases/reuses
slots, not independently reclaimable page allocations. Existing append GPU
success is not evidence of physical grant reduction or driver-memory savings.
No model/TP/standard-runner, native kernel ABI, manager or global registry was
changed. Calling the new allocator is the only opt-in; there is no dispatch flag
that can silently select an unsupported storage mode.

## Contract

- `PageAllocation` owns the full backing grant through shared custody. Adoption
  is an unsafe backend boundary: the root extent, actual grant size, address
  space and ownership must be truthful. A borrowed slice cannot be adopted as a
  fresh allocation. Backend adapters must admit/charge before allocating.
- `PageAllocationIdentity` includes an opaque allocation incarnation, opaque
  binding incarnation and checked nonwrapping generation. Reusing an address,
  logical page ID, byte offset or numeric generation cannot resurrect an old
  chunk. Creating a new binding of a still-live grant is also a new incarnation.
- A page/plane binding covers a checked, aligned byte extent. A chunk is
  page-relative, nonempty and element-aligned; it cannot cross that extent.
  This is not a substitute for schema/layer/frontier/transaction authentication.
- A `PagePlacementCustody<B, F>` belongs **inside the existing Entry**, not beside
  it as another transaction journal. It retains original and staged backing
  grants, deduplicates accounting for shared plane grants, rejects mixed address
  spaces, and rejects every staged/original and staged/staged physical overlap.
  Alias checks use absolute ranges, not just allocation identity or slot numbers.
- `begin` precedes every physical submission. The safe custody methods accept
  only three opaque, non-cloneable `PageCompletionToken`s: producer, DMA and
  final consumer. Each token is bound to the same opaque submission incarnation,
  generation and domain. `PageCompletionToken::after_submission` is an unsafe
  trusted adapter boundary; `PageCompletionFence` is an unsafe implementation
  contract. An arbitrary client ACK, numeric tuple, event from another stream or
  relabeled token cannot become evidence. A compute event alone does not
  establish transfer completion on an unrelated stream.
- `poll` mints one opaque, non-cloneable `PagePlacementProof` only after all three
  submission-domain tokens complete. Proof creation does not itself release
  allocation custody. `release(proof)` releases this record's references;
  `reuse(proof)` instead advances the generation without releasing allocation
  bytes.
- Partial/unrecorded submission, failed or unwinding completion queries retain
  custody. Unknown rejects ordinary polling, late fence installation, new work
  and chunk validation. Cancellation is not a fence. `reap` accepts only three
  separately minted `after_full_drain` tokens. A complete drain moves the
  subordinate record to terminal `Drained`: it permits physical release but
  never chunk validation, new submission or `reuse`. It does not clear the
  existing transaction Entry's sticky Unknown, logical pins or abort/reaper
  state; the parent Entry must independently complete its existing protocol.
- Dropping a submitting/pending/Unknown record intentionally retains backing and
  fence custody by leaking them. This is a fail-closed fallback, not a reaper or
  successful cleanup. Keep the existing Entry alive to make recovery possible.

## Metrics: do not combine different scopes

| Observation | Meaning | Not evidence of |
| --- | --- | --- |
| `slot_reuses` | Number of fixed-plane page/plane extents reused by this custody record | Allocation release or driver free |
| `live_allocation_bytes` | Full granted bytes retained by this record, deduplicated by grant; includes pending and Unknown | Context-wide residency, logical page bytes, or last-owner drop |
| `driver_freed_bytes = None` | No driver-free observation exists in the metadata contract | Zero measured driver frees or positive savings |
| `allocator_metrics().live_granted_bytes` | Existing context-wide allocator live grants | CUDA free memory; pending/cached grants can remain reserved |
| `pending_retirement_bytes`, `reserved_bytes`, `driver_free_bytes` | Existing context-wide retirement/cache/actual driver telemetry | Per-page attribution when other allocations are active |

A record release can set its retained byte count to zero while another owner
still holds the allocation. Even a last-owner drop can enqueue retirement and
leave a segment cached. The opt-in allocator exposes the existing context
metrics without fabricating a page-local driver-free counter.

## Next integration layer (deliberately not implemented here)

1. Admission reserves actual rounded page grants before allocation. Embed the
   resulting `PagePlacementCustody` in the original transaction Entry, alongside
   its logical pins, budget, prefix-read and DMA resources. Charge original and
   staged backing together; never refund on cancel, enqueue, or slot release.
2. The storage/view adapter must authenticate the existing transaction nonce,
   view epoch, logical witness, plane dtype/layer geometry and slot epoch, then
   resolve a `PagePhysicalChunk` into a **borrowed, retained** typed device view.
   Do not return an independent raw pointer or a buffer that outlives Entry
   custody. No such pointer/view API is exported by this allocation-only slice.
3. Either add an explicit independent-page pointer-table kernel contract, or an
   explicit page-local read/partial-attention operation plus a correctly fenced
   merge. The existing base-pointer + i32-slot ABI cannot address arbitrary
   allocations. Do not reinterpret an allocation identity as a kernel slot.
4. Before dispatch, call `begin`; after actual dispatch, bind producer, exact DMA
   tickets/transport evidence and final output-consumer evidence. Bind failures
   to the same Entry's Unknown/reap path, including errors that returned no
   ticket. A public metadata tuple is not proof. No new manager is needed.
5. After physical proof **and** the existing logical publish/abort/retirement
   obligations, release the old grant. Measure allocator live/pending/reserved
   deltas and successful driver-free deltas in a controlled workload. Only then
   claim the specific measured form of memory saving; slot reuse alone never
   qualifies.

## Validation (no GPU execution)

Validation uses CUDA compilation/linking and driver-free metadata/build tests
only (`CUDA_VISIBLE_DEVICES=''`, `NVIDIA_VISIBLE_DEVICES=void`); no GPU execution.
Every validation command had a 900000 ms timeout.

- CUDA 12.4.99, `FERRULE_CUDA_ARCH=sm_86`, from
  `target/validation/qwen35-35b-current/toolchain/env-cuda124.sh`.
- Production branch (`FERRULE_CUDA_TEST_ORACLE=0`): CUDA all-targets clippy,
  all-targets `--no-run`, `cuda_build_paths`, and `host_kv_protocol` passed.
  The 12.4 production compile produced exactly two production CUDA archives.
- Oracle branch (`FERRULE_CUDA_TEST_ORACLE=1`): oracle-only compile and build
  contract tests passed (6 in both branches). The related native manifest test initially exposed
  a missing W8A16 availability stub; the backend test now covers that symbol and
  independently checks the W8 bit.
- Strict CPU: `cargo test --locked -p ferrule-backend --no-default-features
  --all-targets` passed, including 39 unit tests (16 physical-placement tests),
  5 build-contract tests and 23 HostKV tests; doc compile-fail tests passed (6).
- CPU no-default-features all-targets clippy and targeted rustfmt passed.
- No CUDA 13.2/sm90 result is used as compatibility evidence.
- `nm`, `ar`, and CUDA 12.4 `cuobjdump --list-elf` audit both isolated target
  directories: production has no scalar-reference archive or oracle symbols;
  the oracle flag adds the separate scalar-reference archive and the existing
  gated diagnostic symbols. All CUDA images are sm86. Each archive has one
  source TU object plus the toolchain-generated dlink object (not a third
  production source TU). Local audit records:
  `target/validation/hostkv-physical-cuda124-production/SYMBOL_AUDIT.txt` and
  `target/validation/hostkv-physical-cuda124-oracle/SYMBOL_AUDIT.txt`.

Reproduction environment (always override the script's GPU=0 setting):

```sh
. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void FERRULE_NO_CUDA=1
export CARGO_BUILD_JOBS=2 FERRULE_CUDA_TEST_ORACLE=0
nvcc --version # release 12.4, V12.4.99; script sets FERRULE_CUDA_ARCH=sm_86
export CARGO_TARGET_DIR=target/validation/hostkv-physical-cuda124-production
cargo clippy --locked -p ferrule-backend --features cuda --all-targets -- -D warnings
cargo test --locked -p ferrule-backend --features cuda --all-targets --no-run
cargo test --locked -p ferrule-backend --features cuda --test cuda_build_paths --test host_kv_protocol
cargo test --locked -p ferrule-backend --features cuda --lib physical_placement::tests
export CARGO_TARGET_DIR=target/validation/hostkv-physical-cuda124-oracle
export FERRULE_CUDA_TEST_ORACLE=1
cargo clippy --locked -p ferrule-backend --features cuda --lib --test cuda_build_paths --test numeric_w8a16 -- -D warnings
cargo test --locked -p ferrule-backend --features cuda --test cuda_build_paths --test numeric_w8a16 --no-run
cargo test --locked -p ferrule-backend --features cuda --test cuda_build_paths
```

The numeric oracle test binary is compiled/linked only, not run. Build-contract
coverage checks the exact `=1` opt-in, Rust cfg and native define gates, the
third TU's guarded scope, independent source/archive ownership, strict oracle
math flags and absence of scalar-reference dependencies in production sources.
It does not simply increase the allowed global builder count.
