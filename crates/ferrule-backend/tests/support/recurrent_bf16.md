# HF eager BF16 recurrent backend handoff

This is an opt-in backend primitive set, **not** admission of a complete Qwen3.5
35B BF16 model/profile, MoE implementation, fused-vLLM equivalence claim, or a
performance result. Existing F32 recurrence and kernel behavior are unchanged.

## Model-owner API

Import buffer descriptors from
`ferrule_backend::cuda::operators::recurrent_bf16` and shape-only layouts from
`ferrule_backend::cuda::operators::recurrent`:

1. `CudaOperators::causal_depthwise_conv1d_silu_bf16_into`
   accepts `Bf16CausalConv1dBuffers` + `CausalConv1dLayout`.
2. `CudaOperators::gated_delta_net_bf16_into` accepts
   `Bf16GatedDeltaNetBuffers` + `GatedDeltaNetLayout`.
3. `CudaOperators::gated_rms_norm_bf16_into` accepts BF16 input, ordinary BF16
   norm weight, BF16 z gate, BF16 output, shape-only `F32RowsLayout`, epsilon.
   Here the legacy layout name denotes dimensions only, **not storage dtype**;
   rows=tokens*value_heads, width=value_dim. Do not use offset RMSNorm.

Inputs/projections/weights/history/output are `CudaBf16Buffer` (2-byte words).
`A_log` and `dt_bias` match **HF native BF16 loading**; both are BF16 parameters
expanded to F32 during gate evaluation. `state` is `CudaF32Buffer`, physically
4-byte `[value_heads,key_dim,value_dim]`. QKV rows are all Q, then all K, then
all V, with contiguous Q/K head repetition to value heads. Do not interchange
this state layout with a vllm `[value_dim,key_dim]` state.

Use existing explicit GPU row conversion for projection accumulators when
needed. Merely expanding BF16 operands to F32 and calling the old recurrence
would omit intermediate Q/K and beta rounding; it is **not** equivalent.

Each call scans one chronological sequence: rows=1 decode; rows>1 prefill or
continuation from supplied state. Initialize fresh histories/states explicitly;
no call implicitly clears them. Conv shifts history and stores each raw BF16
input before calculating that token; history is `[channels,kernel_size]`,
oldest first. Prefill uses the recurrent equation instead of HF chunked GEMMs,
so reduction order is not claimed bitwise identical to the chunk implementation.

## Storage, scratch, aliases and custody

`Bf16RecurrentStorage::for_layouts` validates compatible conv/GDN layouts and
reports logical persistent bytes:

- conv history: `2 * channels * kernel_size`;
- SSM: `4 * value_heads * key_dim * value_dim`;
- caller-owned activations: raw/conv QKV each `2 * rows * channels`, a/b each
  `2 * rows * value_heads`, core/z/norm output each
  `2 * rows * value_heads * value_dim`;
- weights/optional bias: ordinary BF16 element counts times 2;
- hidden device scratch: **zero** for all three operations.

Allocator alignment/retirement, checkpoints, replicas and simultaneous live
buffers must be budgeted separately. State checkpoints need **both** actual
BF16 history and actual F32 SSM. No F32 allocation is relabeled as BF16.

All extents, context owners (not just ordinals), and mixed-dtype byte ranges are
checked before submission. Read/read overlap is legal; writers must be disjoint
from reads and each other, including checked allocation views. Native entry
points additionally reject invalid dimensions, overflow, null/misaligned
pointers, reserved fields and invalid epsilon before launch. The Rust typed
boundary is responsible for allocation lengths, aliases and custody.

No allocations, copies, synchronization, hidden host math, or state reset occurs
inside these APIs. Each submits one kernel on the owner's compute stream.
Submission is not completion. Keep custody until stream ordering or completion
proof permits reuse/publication; unknown completion does not authorize rollback
or state reuse. Existing allocator fencing/quarantine is unchanged.

## Local source evidence / exact boundaries

Inspected local `transformers/models/qwen3_5/modeling_qwen3_5.py`, Transformers
5.5.3. Its source hash, Torch version and fixture hash are in
`recurrent_bf16_hf.json`.

| Operation | HF eager boundaries implemented |
| --- | --- |
| conv | BF16 input/history/weight; F32 accumulation; materialize BF16 Conv1d output; SiLU; BF16 output |
| Q/K L2 | BF16 square; F32 sum with BF16 reduction output; BF16 `+1e-6`; BF16 rsqrt; BF16 multiply; then expand to F32 |
| Q scale | F32 `1/sqrt(key_dim)` after Q/K normalization |
| beta | BF16 sigmoid(b), then expanded to F32 |
| decay | `-exp(A_log.float()) * softplus(a.float()+dt_bias)` in F32, softplus threshold=20; exp in F32 |
| recurrence | decayed S, memory reduction, correction, S update and Q dot in F32; output materialized BF16 |
| gated RMSNorm | F32 RMS; normalized value BF16; BF16 weight product BF16; SiLU(z.float()) F32; final product BF16 |

Local vLLM source was also inspected, but it is **not** interchangeable:
`model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` allocates `A_log` F32,
uses fused post-conv/L2 helpers, and exposes multiple prefill/decode backends.
`MambaStateDtypeCalculator._mamba_state_dtype` in `mamba_utils.py` chooses conv
state from cache/model dtype, and **auto SSM follows conv dtype** in this local
version. Therefore a matching BF16-history/F32-SSM comparison must explicitly
select F32 SSM cache, not assume vLLM `auto` does so. Its `fused_gdn_gating` uses
F32 g and beta storage matching b dtype. In contrast, the inspected prefill
`third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py` performs
Q/K squares, reductions and inverse norm in F32, casts normalized Q/K only at
store, and allocates **both g and beta in F32**. Its `causal_conv1d.py` computes
SiLU on the F32 accumulator before output storage. These fused paths omit HF
eager intermediate boundaries and must not be silently substituted for this API.

## Independent oracles and scope of validation

`recurrent_bf16_hf.py` creates a deterministic tiny Qwen3.5 checkpoint, saves it,
then **loads it through HF `from_pretrained(dtype=torch.bfloat16)`**, asserting
parameter dtypes. It calls native HF torch conv-update, recurrent rule, chunk
rule and gated norm, without importing production code or transplanting kernel
math. The fixture contains 1/7/65/129-token cases, zero/nonzero initial states,
bias/no bias, repeated heads, extreme gates, and per-token history/SSM captures.

A discovered Torch CPU quirk matters: short BF16 rsqrt scalar tails double-round
in this build (`rsqrt(BF16(0.00811767578125))` gives 11.0625 rather than the
single-round result 11.125). The generator selects AVX512 and repeats eight
identical **independent batches** to eliminate scalar tails, verifies the native
rsqrt probe, and retains one batch. No HF math function is replaced. Re-running
this generator requires that CPU capability and the recorded dependencies.
This is a CPU-generated native-HF boundary oracle, not a claim that an actual
HF CUDA run was performed. Transcendental/reduction tolerances remain explicit.

`recurrent_bf16_reference.rs` is a separate scalar CPU check: independent
nearest-neighbor BF16 rounding, no production logic includes. Tests check captured
conv/QK/beta/norm boundaries and F32 states, and reject deliberately omitted
boundaries or BF16-rounded SSM. GPU tests consume the HF captures directly,
isolate stages, scan whole/1/3/64-token segments and compare intermediate state.
Additional ignored tests validate every owner/extent and real allocation aliases.

Validation performed with GPUs hidden, CUDA 12.4.99 and `FERRULE_CUDA_ARCH=sm_86`:

- CPU HF fixture generation and independent CPU tests;
- native CUDA compilation / backend `cargo check --features cuda`;
- dedicated CUDA-linked tests: CPU tests and prelaunch guards run;
  actual GPU numerical/owner/alias tests remain ignored.

GPU owner must run the ignored tests before claiming GPU numerical acceptance:

```sh
cargo test -p ferrule-backend --features cuda --test cuda_recurrent_bf16 -- --ignored
cargo test -p ferrule-backend --features cuda --test cuda_recurrent_bf16_validation -- --ignored
cargo test -p ferrule-backend --features cuda --lib recurrent_bf16 -- --ignored
```

No GPU/performance execution, backend KV/transport changes, W8 numeric changes,
model wiring or complete Qwen3.5 profile declaration is part of this patch.
