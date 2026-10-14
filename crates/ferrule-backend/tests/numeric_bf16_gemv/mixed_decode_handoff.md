# Mixed decode reuse — backend API handoff

API implemented and ready for model integration; GPU numeric validation deferred.
All types are exported by `ferrule_backend::cuda::operators::linear`.

```rust
impl CudaBf16GemmEx {
    pub fn decode_numeric_bf16_into(
        &mut self,
        op: &CudaOperators,
        weight: &CudaNumericFp8Artifact,
        target: &mut CudaBf16Buffer,
    ) -> ferrule_common::Result<()>;

    pub fn decode_numeric_bf16_pair_into(
        &mut self,
        op: &CudaOperators,
        weights: [&CudaNumericFp8Artifact; 2],
        target: &mut CudaBf16Buffer,
    ) -> ferrule_common::Result<()>;

    pub fn execute_bf16_pair_view(
        &mut self,
        op: &CudaOperators,
        plan: Bf16GemmExPlan,
        weight: &CudaBf16Buffer,
        input: &CudaBf16Buffer,
        outputs: [&mut CudaBf16Buffer; 2],
        layout: Bf16GemmLayout,
    ) -> ferrule_common::Result<()>;
}
```

- Decode uses the existing prepared owner (no extra cuBLAS handle). Target must
  contain exactly `N*K` BF16 words. Numeric layout: complete 128x128 blocks,
  zero origins, BF16 scales, same admission as existing numeric GemmEx. No
  host read, allocation, synchronization, activation packing, or GEMM.
- Decode calls the original numeric decode implementation:
  `BF16_RNE(__fmul_rn(fp8_value(byte), bf16_scale))`, not W8A16 arithmetic.
- To populate gate/up, allocate exactly `2*N*K` BF16 words and call
  `owner.decode_numeric_bf16_pair_into(op, [gate, up], &mut target)`.
  Order is **gate rows, then up rows**. No model-side offsets/slices needed.
  Both source layouts must be identical (including BF16 scale dtype, zero
  origins, full scale blocks), on the owner's context; the operator must be
  the exact prepared stream owner. Both sources and the entire target are
  preflighted before the first launch. Target must not overlap either source's
  weights/scales or provider storage; short/oversized targets are rejected.
  Two original decoder launches, no `2NK` temporary, copy or synchronization.
  Decode is M-independent and does not require a separate warmed decode view.
- Pair accepts the existing warmed **full `2N` plan**, not a half-width plan.
  `layout = Bf16GemmLayout::contiguous(M, 2*N, K)`. Gate/up widths are equal,
  matching the original pair API. Input is exact packed `[M,K]`; each output
  is exact packed `[M,N]`. Strided/padded operands are rejected, not packed.
- Pair calls the same GemmEx geometry and original bit-preserving `split_pair`.
  Only intermediate `[M,2N]` output uses existing provider staging; weights
  and activations are passed directly. Counters: single decode 1, pair decode 2,
  resident single 1,
  resident pair 2 (old cold single 3 / pair 5 unchanged).
- Eager-only, exact stream owner, context/extent/alias/capability preflight;
  no workspace rebind. Existing submission failures poison owner + mark
  terminal Unknown. Caller retains artifacts/target/operands to last consumer,
  including quarantine on Unknown, exactly as with existing shape views.
- Model/runtime/composition/paper are not changed. GPU execution deferred.

## Validation (2026-10-10, GPU hidden)

- CPU tests: `bf16_gemmex` (3), `numeric_bf16_gemv` (1),
  `numeric_w8a16` (4), `numeric_fp8_payload` (1): all passed.
- CUDA 12.4.99 / `sm_86`: backend library tests compiled and linked.
- `cargo test -p ferrule-backend --features cuda --lib mixed_decode_reuse_tests`:
  3 host-only native preflight tests passed; 2 GPU tests ignored.
- CPU and CUDA `cargo clippy -p ferrule-backend --lib --tests -- -D warnings`
  passed (CUDA invocation adds `--features cuda`).
- Scoped rustfmt and diff whitespace checks passed.
- GPU tests in `mixed_decode_reuse.rs` cover cold original decode vs explicit
  destination bytes, cold/native/resident single and pair bit equality at M1/M5,
  full N/K = 1024/2048, 8192/2048, 2048/512, launch/copy/allocation/fence
  counters and owner/extent/stride/alias/capture/poison rejection. These tests
  have **not** run on a GPU; no GPU numeric equivalence claim is made yet.

Implementation is restricted to the BF16 GemmEx operator, its existing private
provider/FFI glue, `bf16_gemmex.cuh`, the shared decoder's optional destination
in `bf16_linear.cuh`, and dedicated backend tests/handoff. Existing single,
cold pair, W8A16, model, runtime, composition and paper behavior is unchanged.

Caller must account for resident target allocation separately from the existing
prepared provider storage grant. No automatic cache, fallback, or graph API is
introduced. Reuse only after same-stream decode submission and retain the
allocation through all consumers; do not recover or reuse after Unknown.

## Pair decode follow-up

`decode_numeric_bf16_pair_into` is implemented; no model files changed.

- Native pair preflight covers both sources, matching geometry/streams, exact
  adjacent target halves, and whole-target overlap with all four source spans.
  Immutable source/source aliasing is allowed. Validation completes before
  either launch. Both launches reuse `ferrule_bf16_gemmex_decode_launch` and its
  original `decode_weights` body; a launch failure uses existing owner Unknown.
- M1/M5 GPU tests now exercise the public pair-decode API against whole-2N
  single decode, original cold decoder bytes, and original paired GemmEx exact
  output. Host payload vectors retire before decoding; uploaded native BF16
  host words retire before execution. After all encoded reads have a completion
  proof, encoded artifacts retire and the resident pair repeats both shapes.
- Follow-up validation with `CUDA_VISIBLE_DEVICES=''` and
  `NVIDIA_VISIBLE_DEVICES=void`: CPU 9 tests passed; CUDA12.4/sm86 compile/link
  passed; 3 native host-only preflight tests passed, 2 GPU tests ignored;
  CPU/CUDA clippy (`--lib --tests -- -D warnings`), scoped rustfmt and whitespace
  checks passed. GPU numeric tests remain unexecuted.

Model call (the target must already be allocated and separately accounted):

```rust
owner.decode_numeric_bf16_pair_into(op, [gate, up], &mut target_2nk)?;
// Same stream, same original prepared full-2N plan and packed input:
owner.execute_bf16_pair_view(
    op, plan_2n, &target_2nk, input, [gate_output, up_output],
    Bf16GemmLayout::contiguous(m, 2 * n, k),
)?;
```

No public raw pointer or model-side slice is required. Decode does not itself
retire the encoded sources: preserve their existing last-consumer/Unknown
custody until completion is proven.

## Typed BF16 arena view follow-up

API implemented and validated with GPU hidden, available directly on the
existing `CudaBf16Buffer` alias exported by `cuda::operators::linear`:

```rust
impl CudaTypedBuffer<u16> {
    pub fn slice(&self, offset: usize, len: usize) -> ferrule_common::Result<Self>;
}
// Element units, not bytes; offsets are relative to this view:
let gate = arena.slice(gate_offset_words, gate_len_words)?;
```

This is a shared allocation view, not an exclusive mutable borrow or a copy.
The existing `DeviceBuffer::slice` retains the original allocation/context even
if the root or an intermediate view is dropped. Bounds and element/byte/address
overflow are checked. Zero length is allowed at offsets `0..=self.len()` just
like `DeviceBuffer::slice`. No allocation/copy/launch/synchronization is added.
Caller must not concurrently write overlapping aliases or race writes with
reads. Preserve the existing same-owner stream ordering, last-consumer and
Unknown rules; a fresh `&mut` wrapper does not establish allocation exclusivity.

Typed-view validation:

- `cargo test -p ferrule-backend --no-default-features --test cuda_bf16_slice`:
  3 passed (bounds/empty/nested extents, u16 multiply/add/address overflow,
  structural guard for existing DeviceBuffer Arc allocation/context retention).
- CUDA12.4/sm86 hidden `cargo test -p ferrule-backend --features cuda
  --test cuda_bf16_slice`: compiled/linked, same 3 host tests passed; 1 GPU
  lifetime microtest ignored. The CPU custody guard is structural, not a
  simulated or executed device-lifetime proof.
- CPU targeted clippy, CUDA `--lib --tests` clippy with `-D warnings`, and scoped
  rustfmt check passed.
- Pending authorized GPU microtest checks nested pointer byte offsets, shared
  context, root/intermediate drops with a surviving readable child, unchanged
  allocator requests/live allocation while retained, and zero allocation/copy/
  kernel/sync counters for slice creation and nonfinal view drops.

Only `operator_owner.rs`, the host-only operator range helper,
`tests/cuda_bf16_slice.rs`, and this handoff changed for typed views.
No model/runtime/paper changes and no new unsafe code.

## Allocation-only dedicated BF16 arena (root-before-init P1)

API implemented and validated with GPU hidden; ready for model integration:

```rust
impl CudaOperators {
    pub fn allocate_bf16_arena_uninitialized(
        &self,
        len: usize,
        grant_budget_bytes: usize,
    ) -> ferrule_common::Result<CudaBf16Buffer>;
}
```

`len` is BF16 words. Budget is **physical allocator grant bytes**, not logical
word bytes: use the already-admitted arena grant, or calculate it with
`ferrule_backend::cuda_allocation_grant_bytes(len.checked_mul(2)?)`.
This checks the grant bound, not a new reservation; caller retains its existing
budget claim. Dedicated backing avoids unrelated shared-segment slack.

Allocation only: existing `DeviceBuffer::<u16>::uninitialized_dedicated` on the
operator's context, existing allocation counters/failpoints, checked length and
rounding/budget, capture and terminal Unknown rejection. No memset, decode,
copy, kernel, new handle, explicit sync, or initialized-data promise.
Zero words are allowed with zero grant. Allocation failure submits no arena
initialization; existing allocator retirement protection is unchanged.

**Model change required (not made by backend):** replace the root
`zero_bf16_buffer` with this allocation-only call, immediately install the
returned root in the owner's retained arena state, then obtain typed slices
and submit fixed/slot decodes. Every range must be completely written before
it is marked ready or consumed. Do not zero the whole ~3 GiB arena. Install
custody before the *first* initialization launch; keep root/claims through
Unknown even if a decoder partially writes and then fails. Keep slot readiness
separate from allocation. This addresses root-before-init ordering, not a
claim that the previous allocator retirement path was a UAF.

The API returns opaque device storage, not a host reference or a proof of
initialization. Like existing uninitialized scratch allocation, write-before-
consume is a caller obligation, including before downloads. No new unsafe
block, unchecked constructor, or raw-pointer escape is introduced. Shared
slice/overlap/last-consumer rules above still apply.

Allocation-only follow-up validation:

- `cuda_bf16_arena`: 2 CPU tests passed (word/rounding overflow, exact grant
  budget including the 3 GiB case; allocation-only call-chain/source guard).
- `cuda_bf16_slice`: 3 CPU regressions passed.
- CUDA12.4/sm86 hidden compile/link and the same 5 host tests passed; 2 arena
  GPU microtests plus the existing slice GPU microtest remain ignored.
- CPU targeted clippy, CUDA `--lib --tests -- -D warnings` clippy, and scoped
  rustfmt checks passed.
- Pending GPU arena microtests cover dedicated physical-grant accounting,
  installed root before exact fixed/slot decode, no read before initialization,
  alias lifetime, budget/overflow/capture/allocation-failpoint/shutdown errors,
  and allocation-without-kernel/copy/sync counters. The CPU source guard also
  checks the runtime delegation excludes `zeroed`/memset/initialization on
  success and error; counters alone cannot prove absence of an uncounted native
  memset. No GPU trace or execution is claimed for this follow-up.

This follow-up changes only `src/cuda/operator_owner.rs`,
`src/cuda/operators/bf16_arena.rs`, `tests/cuda_bf16_arena.rs`, and this handoff.
The model must still apply the replacement and root-before-init install order.
No model, runtime, native kernel, or paper files were changed for this API.

## Bounded pair-SwiGLU follow-up

The same prepared full-2N owner now provides
`execute_pair_swiglu_view_into` (original FP8 pair) and
`execute_bf16_pair_swiglu_view_into` (contiguous resident BF16 pair), writing
caller-owned `[M,N]` activation directly. Original pair APIs remain available.
See [pair_swiglu_handoff.md](pair_swiglu_handoff.md) for the exact signatures,
rounding/ownership contracts, CPU-only validation and pending GPU selectors.
