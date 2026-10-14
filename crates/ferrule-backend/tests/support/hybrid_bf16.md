# Backend HF eager BF16 gate and MoE combine

Implemented native kernels and typed APIs, not full-model admission. Existing F32
and legacy BF16-scatter/W8 behavior and names are unchanged. No new operator family
or provider framework: gate extends `DataArgs` kind 16, combine extends `MoeArgs`
kind 14, with existing native entrypoints/translation unit.

## Exact model-facing signatures

```rust,ignore
use ferrule_backend::cuda::operator_owner::{
    CudaOperators, CudaBf16Buffer, CudaF32Buffer, CudaI32Buffer,
};
use ferrule_backend::cuda::operators::standard_bf16::Bf16GateLayout;

// Methods on CudaOperators:
pub fn sigmoid_gate_bf16_into(
    &self,
    input: &CudaBf16Buffer,
    gate: &CudaBf16Buffer,
    output: &mut CudaBf16Buffer,
    layout: Bf16GateLayout,
) -> ferrule_common::Result<()>;

pub fn weighted_combine_bf16_into(
    &self,
    values: &CudaBf16Buffer,
    route_rows: &CudaI32Buffer,
    weights: &CudaF32Buffer,
    output: &mut CudaBf16Buffer,
    output_rows: usize,
    width: usize,
) -> ferrule_common::Result<()>;
```

- `Bf16GateLayout::Elementwise { rows, width }`: input, gate and output are
  exactly `[rows,width]`.
- `Bf16GateLayout::RowBroadcast { rows, width }`: input/output are exactly
  `[rows,width]`, gate is exactly `[rows,1]`.
- Gate computes `BF16_RNE(input * BF16_RNE(sigmoid(F32(gate))))`, with no cutoff.
  This is not SwiGLU or the DeltaNet SiLU gated norm.
- Combine input is exactly `[route_rows.len(),width]` **BF16 projections**.
  Convert F32 GEMM scratch per route to BF16 before this call. Weights are exactly
  `[route_rows.len()]` F32. Output is exactly `[output_rows,width]` BF16.
- Combine overwrites output from zero and folds matching routes in supplied
  order: `contribution=BF16_RNE(F32(projection)*weight_f32)`, then
  `accumulator=BF16_RNE(F32(accumulator)+F32(contribution))`. No atomics or FMA.
  HF eager requires routes packed in ascending **expert index**, not top-k rank.
  Duplicate target rows are supported; negative/out-of-range routes are masked.
  Unrouted output rows are zero. Zero routes/dimensions are rejected.
- All operands must belong to the exact context owner, not merely the same GPU.
  Exact extents, checked products/index bounds and writable byte-range alias
  checks precede submission. Read/read alias is permitted; output is disjoint
  from every read, including F32 weights and I32 metadata.
- Each API submits one kernel on the existing owner compute stream. No scratch,
  allocation, copy, synchronization, persistent mirrors or additional grants.
  Existing allocation grants and stream-fenced retirement remain authoritative.
  Success is submission, not completion. The caller retains buffers and fences
  cross-owner reuse; failed submission is not a rollback guarantee.

Router arithmetic is **not changed**: native HF logits are BF16 and full-softmax,
selected weights and renormalization are F32. Do not apply the blanket legacy
BF16 router-weight rounding policy. The existing selected-logit-softmax API is
not claimed to reproduce HF's full-softmax F32 operation graph/tie behavior.

## Independent evidence and CPU contracts

`hybrid_bf16_hf.py` reuses only the model handoff's checkpoint loader and passive
observer. Expected tensors are native HF attention/shared-gate/expert results;
no HF expert forward algorithm or Ferrule kernel is used to produce expected
outputs. Captured down projections, route metadata and native expected output
are saved in `hybrid_bf16_hf.txt`; package/source/fixture hashes are in the JSON.
No backend test dependency is added. Rust tests read the text directly.

```sh
CUDA_VISIBLE_DEVICES= /opt/conda310/bin/python crates/ferrule-backend/tests/support/hybrid_bf16_hf.py
CUDA_VISIBLE_DEVICES= cargo test -p ferrule-backend --no-default-features --test cuda_hybrid_bf16
```

CPU contracts exactly replay all 128 elements in each native boundary capture:
attention old single-rounding differs at 38/128 (max 0.0078125), shared gate
43/128 (max 0.00390625), old F32 MoE accumulation 67/128 (max 0.00390625).
Scalar gate: `0.515625,-0.9375 -> 0.14453125`, not `0.1455078125`.
No tolerances are used or widened. These are CPU boundary-policy checks, **not
GPU equivalence or full-model numerical acceptance**.

## GPU tests compiled, not executed

Do not execute until a sole GPU owner assigns the run. Four ignored tests exist:

In `--test cuda_hybrid_bf16`:
- `gpu::sigmoid_gate_matches_native_hf_elementwise_and_row_broadcast`
- `gpu::weighted_combine_matches_native_hf_and_masks_routes`
- `gpu::every_operand_owner_and_exact_extent_rejected_before_submission`

In `--lib`, private access to real allocation views:
- `cuda::operators::standard_bf16::tests::writable_aliases_rejected_and_read_aliases_keep_grants`

They cover full same-input native expected tensors, the scalar counterexample,
odd widths, masks with NaN payloads, overwrite semantics, all operand owners and
short/long extents, exact/partial writable aliases, read aliases, no hidden
allocations/copies, unchanged grants and capture-safe submission.

Non-ignored CUDA-feature tests are CPU-only layout/native preflight checks.
Invalid native descriptors return before any launch or CUDA context acquisition.

```sh
. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
export CUDA_VISIBLE_DEVICES=
cargo test -p ferrule-backend --features cuda --test cuda_hybrid_bf16 --no-run
cargo test -p ferrule-backend --features cuda --lib standard_bf16::tests --no-run
```

CUDA12.4.99 / sm_86 hidden-device compilation is separate from GPU numerical
validation. No GPU test, benchmark, probe or ownership acquisition was performed.
