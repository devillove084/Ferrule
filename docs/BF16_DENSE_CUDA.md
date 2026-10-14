# Standard dense BF16 CUDA profile

`StandardDecoderSegment::prepare_cuda(resources, plan,
ExecutionPrecisionPolicy::bf16_compatibility(), max_positions,
max_parameter_bytes, ops)` prepares the real BF16 path. Select a matching
`StandardGqaPlanes` with `KvElementType::Bf16` and use the existing
`PagedKvBackend`, entered `CudaKvView`, `execute_cuda_bound`, leave and commit /
rollback protocol. The segment exposes its actual `precision()`,
`activation_dtype()` and `kv_element_type()`.

`Bf16DenseCudaProfile::validate(resources)` is available without CUDA for early
admission. This first profile is **dense ungated GQA only**, ordinary RMSNorm,
unclamped SwiGLU, bias-free K8 linear projections, split-half prefix RoPE without
scaling, and native unscaled BF16 checkpoint parameters. GDN, gated GQA, offset
norms, MoE, numeric FP8, TP and HostKV chunked append are not admitted by this
standard dense profile. Native dense Qwen3 serving now exposes
`serve --model-precision bf16` through the existing single-stage pipeline;
[serving selection and budgets](../crates/ferrule-composition/tests/dense_bf16/README.md)
are maintained separately. This is not the Qwen3.5 numeric FP8 Mixed profile.
Existing F32 defaults and `W8A16Bf16F32Accumulate` behavior/naming are unchanged.

For a native BF16 dense Qwen3 checkpoint (not a quantized or Qwen3.5 export):

```sh
ferrule serve /path/to/Qwen3 --backend cuda --model-precision bf16 \
  --engine auto --devices 0 --max-tensor-mb 2048 --kv-cache-mb 1024
```

This selects a supported construction path, not a numerical acceptance result.
See [current acceptance](CURRENT_ACCEPTANCE.md); Mixed quality receipts cannot
upgrade this separate profile's GPU quality status.

## Physical storage and arithmetic

- Resident checkpoint parameters: **2 bytes/element**, canonical-ID deduplicated;
  no resident F32 weight mirror. BF16 checkpoint values are losslessly staged via
  the existing materializer and uploaded as BF16 words. This does not claim
  bounded host checkpoint materialization.
- RoPE cosine/sine tables: **2 bytes/element**; equal tables share one binding.
- Activations, residuals, gate/up/product and logits: actual `CudaRows::Bf16`,
  **2 bytes/element**. F32 row inputs are rejected, not relabeled.
- K and V cache planes: actual BF16 words through the existing authenticated BF16
  transaction guard. Payload bytes are `2 * slots * layers * page_tokens *
  kv_heads * head_dim` **per plane**, excluding allocator/metadata overhead.
  COW, rollback, commit and preemption use the existing custodian/journal.
- GEMM: backend native BF16 operands, F32 accumulation/output scratch followed by
  explicit BF16-RNE writeback. Missing kernel support is an error, not fallback.
- RMSNorm: F32 variance and normalization; cast normalized value to BF16 before
  BF16 weight multiplication and final writeback (Qwen3/Llama semantics).
- RoPE: BF16 tables and inputs, separately rounded multiply tensors and final
  addition; no single fused FMA boundary.
- SwiGLU: BF16 linear results, one SiLU tensor boundary, a separate product
  boundary, then BF16 down projection. Do not round internal exp/div operations.
- Eager GQA: BF16 dot-result and scaled-score boundaries; F32 stable softmax,
  BF16 probabilities, F32 value accumulation, BF16 attention result. The backend
  keeps its existing F32 Q/output ABI: Q is an exact temporary expansion of BF16,
  and output is rounded before it is exposed as model rows. Legacy BF16 online
  attention and F32 attention are unchanged.
- Host/PP/logits export: exact F32 expansion of BF16 values through existing
  pinned transfer slots. HostRows retains its existing expanded-F32 BF16 contract;
  F32 staging allocations are temporary, not model state.

`resident_parameter_bytes()` includes BF16 weights and device RoPE tables. The
small two-layer test image additionally asserts this count exactly. F32 GEMM/Q/
export scratch and allocator slack are not part of that resident-parameter count.
The path is synchronous/correctness-first; it is not a speed or memory-budget
admission claim. BF16 operations reuse the ordinary standard forward loop and
completion barriers rather than creating another manager or forward program.

## Evidence and limitations

`scripts/standard_bf16_reference.py` generates a no-download, CPU Hugging Face
Qwen3 eager oracle with deterministic native BF16 weights, every layer hidden
output, all logits and all K/V elements for prefill, decode, forked decode and
continuation. The checked-in fixture records PyTorch/Transformers versions.

- `cargo test -p ferrule-model --test standard_bf16_precision`: CPU admission,
  rejection and independent PyTorch semantic-boundary checks.
- `cargo test -p ferrule-model --features cuda --test standard_bf16_cuda --no-run`:
  compile the ignored all-layer GPU test.
- Run the GPU test explicitly with `-- --ignored` on supported hardware. It checks
  physical BF16 rows/cache, resident bytes, hidden/logit/KV comparisons, COW source
  immutability, restore and rollback. Tolerance is `0.00025 + 0.012 * abs(reference)`;
  no bitwise GEMM/reduction parity claim.

The original GPU-hidden implementation run did **not** establish GPU numerical
acceptance. This documentation update adds no new acceptance for this profile. The legacy CPU BF16 whole-model diagnostic is retained but ignored:
its RoPE currently lacks the separate HF BF16 product boundaries, so it is not
used to certify this CUDA path. This dense subset must not be reported as fair
full-Qwen3.5 BF16 execution. Native dense serving admission now exists as linked
above; numeric FP8 Mixed/GDN/MoE use separate profile and evidence contracts.
