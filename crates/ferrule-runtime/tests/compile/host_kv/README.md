# Ordinary append-attend integration: blocked, not enabled

The current `LiveCommittedGqa` is a bounded read-only DMA operation over an
exclusive, resident committed F32 snapshot. It does not evict pages or reduce
authoritative resident KV storage. No ordinary decoder dispatch selects it.

## Prerequisite progress

A transaction-derived prerequisite now exists: `KvPageManager::derive_append_prefix`
borrows the live reservation, and an internal CUDA append-prefix capability holds
its subordinate record in the existing transaction Entry. It checks immutable
original vs shadow/COW mappings and retains fence custody. The partial contract
merges unnormalized states, not normalized outputs. Production dispatch and actual
append DMA remain disabled. See
`crates/ferrule-backend/tests/host_kv/TRANSACTION_PREFIX.md` for the implemented
boundary, validation, and remaining bridge/budget/DMA work.

## Exact dispatch and custody blockers

1. `ferrule-model/src/transformer/standard/mod.rs::execute_gqa` computes the
   current Q/K/V, applies RoPE, calls `paged_gqa`, then consumes its `Rows` in
   the output projection. The current token must participate in attention.
2. `standard/cuda.rs::paged_gqa` fences `BarrierReason::KvPublish`, then calls
   `KvView::append_and_attend_cuda`. The standard path binds an already-entered
   transaction using `cuda/kv_binding.rs::CudaStandardKvBinding`.
3. Runtime `cache/kv_read_authority.rs::validate_committed_source` rejects an append's
   pending reservation (`TransactionActive`). The reverse order also fails:
   `KvPageManager::reserve` checks committed readers first. A page-aligned
   append or a smaller old-prefix read range does not exempt the transaction.
4. Model `decoder/kv/cuda.rs::committed_read` requires quiescent ownership.
   Backend `kv/committed_read.rs` rejects any active/prepared transaction and
   validates resident mappings, not the transaction's shadow/COW mappings.
   Leaving a transaction is not enough: its reservation/journal remains live.
5. `committed_gqa` requires `custody.query_len == 0` and an unchanged frontier.
   `CudaHostPlacement::new` requires `query_start < committed_tokens`, whereas
   ordinary one-token decode queries at `committed_tokens`. First prefill has
   no nonempty committed source at all.
6. `CommittedGqaOutput` contains normalized rows. Prefix rows cannot be averaged
   with tail rows: composition needs `(max, denominator, numerator)` partials
   for disjoint ranges of the same query/mask, then exactly one normalization.
7. `LiveCommittedGqa` borrows the runtime manager, model backend, and sequence.
   The model's `KvView` has none of these runtime references; making the model
   import runtime would invert the production dependency direction. The live
   adapter cannot simply be called from `CudaStandardKvBinding`.

## Required hook / ownership work before implementation

Do not remove the above guards or commit a half-executed layer to satisfy them.
The append reservation and physical transaction must remain the sole authorities.

- Derive a restricted read capability from the existing logical reservation and
  exact physical transaction/nonce. Bind it to declared read sets, shadow/COW
  slot epochs, layer, frontier and sequence incarnation. DMA and attention
  consumers become subordinate custody of that same transaction, preventing
  commit/abort/recycling until its exact fences are proven.
- Adapt the existing placement engine (not a second host-KV ledger) to consume
  that capability and combine committed-prefix and provisional-tail partials.
  Preserve the ordinary resident path when the admitted HostKV mode is not used.
- **Coordination hook in excluded `standard/cuda.rs::paged_gqa`:** accept the
  transaction-derived attention operation, submit after the existing producer
  barrier, and hand its charged output to the existing consumer lifetime.
  `standard/cuda/kv_binding.rs::append_and_attend_cuda` needs the corresponding
  transaction-bound request path, rather than a second pool/read acquisition.
- The runtime/decoder transaction owner must attach one shared bounded placement
  budget and retain cancel/reap state. The model must receive an opaque capability,
  not access `KvPageManager` directly. The output projection must retain/release
  its output charge after the last consumer fence; returning bare `Rows` cannot
  silently refund it. Unknown retains both transaction custody and credits.
- Synchronous per-layer completion is sufficient initially; adding an async
  forward continuation is not required merely to make the DMA real.

The excluded files were inspected but not edited. No production append integration
or GPU execution is claimed by these contract tests.

## Observability

Enable debug events for `ferrule::host_kv` through the application's tracing
subscriber. `committed_read_rejected` reports a typed logical rejection before
registration. The live adapter reports `committed_gqa_begin`, `cancel`, `unknown`,
`reap`, and `finish` (all with the `committed_gqa_` prefix). Polls do not emit on
Pending; Unknown is emitted only on its first observed transition. Reap success
is not logical release or a budget refund. Finish includes CPU/GPU chunk counts
and whether the output still retains its charge. Begin reports the shared
budget's current total charge, not per-operation usage or measured RSS/VRAM.
No event is an eviction claim, and no Q/K/V values or raw pointers are logged.

## Focused validation (no GPU execution)

- `CUDA_VISIBLE_DEVICES='' cargo test --locked -p ferrule-backend --test host_kv_protocol`
- `CUDA_VISIBLE_DEVICES='' cargo test --locked -p ferrule-runtime --lib append_contract`
- With the existing CUDA build environment and `FERRULE_CUDA_ARCH=sm_86`, use
  `CUDA_VISIBLE_DEVICES='' cargo test --locked --release -p ferrule-runtime --features cuda --doc LiveCommittedGqa`.
  The attached examples are exclusively `no_run` / `compile_fail`; they do not
  create contexts, run kernels, or transfer payloads. The positive example ensures
  the negative borrow tests are not merely passing due to missing CUDA exports.
