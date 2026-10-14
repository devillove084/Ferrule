# Executable native BF16 hybrid model slice

The former sigmoid/combine backend blocker is resolved. The existing standard
hybrid model loop now calls the typed BF16 backend operators. This is **native
BF16, single owner, resident experts**, not numeric-FP8 35B or a new CLI profile.
No backend kernels, common completion types, topology/capacity planner or W8
numeric implementation is changed by this model slice.

## Actual computation/storage chain

`GenericDecoderRunner<HybridCudaDecoder>::hybrid_cuda` selects the native path
when `GenericDecoderOptions` contains `ExecutionPrecisionPolicy::bf16_compatibility()`
and complete bound resources pass `Bf16HybridCudaProfile::validate`.

The same `CudaHybridModule`, `execute_standard_layer_inner`, generic transaction
journal, `CudaHybridSequenceLifecycle` and typed CUDA KV pool remain authoritative:

- Native BF16 checkpoint words -> BF16 resident weights, including every routed
  expert before readiness. No device F32 weight mirror or numeric scale payload.
- Embedding, hidden/residual, projections and device LM-head output: BF16.
  BF16 GEMM inputs/weights, F32 accumulator scratch, explicit BF16 output cast.
- Qwen3.5 offset input/QK/output norms: F32 norm + offset affine, one final cast;
  Qwen3 ordinary BF16 normalization remains its distinct existing branch.
- Head-interleaved query/gate split: exact GPU F32 packing work, BF16 outputs.
  RoPE inverse frequencies/trigonometry are F32; tables/products/output BF16.
- GQA uses the existing typed BF16 append/attention transaction. Q/output F32
  scratch is transient; persisted K/V, gated attention output and residual are
  BF16. Attention/shared gates use the new two-boundary sigmoid operator.
- GDN projections/history/core/gated output: BF16; chronological conv and GDN
  backend operators; persistent SSM: F32. Sequence cohort frontiers and owners
  are checked before mutation; positions advance only after completion proof.
- Router logits: BF16; GPU normalized selected-softmax and route weights: F32.
  This is algebraically equivalent to HF full-softmax + renormalized top-k, but
  not bitwise the same reduction; ties select lower expert index. The native HF
  oracle retains the independent full-softmax path and original F32 weights.
- Routed/shared SwiGLU outputs are actual BF16 CUDA allocations. Routes are
  concatenated in ascending expert index and `weighted_combine_bf16_into` rounds
  each F32-weighted contribution and each accumulation to BF16. Shared addition
  and residual addition remain BF16. No F32 model-boundary fallback.
- Public logits use the existing F32 host export of BF16 device logits.

F32 exact-expansion buffers for Q/split/packing/export and F32 norm/GEMM scratch
are explicitly temporary. GDN input gathering and lifecycle history copying use
the existing BF16 row-gather kernel (raw word copies, including signed zero and
NaNs). No persistent F32 conv-history shadow is allocated.

## Model factory and admission

Use the existing sequence:

1. Bind native unscaled BF16 `BoundDecoderResources` and validate
   `Bf16HybridCudaProfile` (dense or shared/routed SwiGLU, bias-free K8 GEMMs,
   at least one full-attention layer, no attachment/partial image).
2. Create ordinary `GenericDecoderOptions::standard_cpu(...)` **as the existing
   descriptor/capabilities constructor**, with BF16 precision. No CPU forward
   is selected; CPU hybrid construction continues rejecting BF16 execution.
3. Call `HybridCudaMemoryEstimate::for_resources(&resources, &options, physical_pages)`.
4. Create `HybridCudaDevice` with that weight/workspace budget and enough state
   bytes for default + committed + working copies + desired forks; use at least
   `(2 * max_sequences + 1) * per_sequence_state_bytes`.
5. Call `GenericDecoderRunner::<HybridCudaDecoder>::hybrid_cuda(resources,
   tokenizer, options, device)` and use the existing public transaction methods.

The estimate counts raw two-byte native weights (canonical aliases once), BF16
rotary tables, physical BF16 KV pages including caller-supplied shadows, BF16
rows and F32 work buffers, route concat/metadata, transient history clone and
allocator/staging headroom. Persistent GDN state per slot is exactly
`2*channels*kernel + 4*value_heads*key_dim*value_dim`. Checkouts/forks/reset consume
the existing hard state cap. Unknown-completion retention keeps both typed state
and charges; no second lifecycle, runner or journal is introduced.

`CudaHybridModule::bf16_profile()` exposes immutable native profile metadata;
`Bf16HybridCudaProfile` separates weight/scale/Rows/KV/conv/SSM/router-weight dtypes.
The existing F32 lifecycle/rank paths and default schema methods remain F32.

External BF16 routing, numeric FP8 weights and HostKV/idle-reclaim are rejected
for this profile. The legacy F32 `ExternalExpertCustody`/transport adapter is not
silently used for BF16 output. No distributed factory or numeric-profile base
was modified. Public **runtime/CLI Qwen3.5 admission stays guarded**; this slice
uses the existing public *model* factory, not a second family runner or an
unvalidated Qwen3.5 serving flag.

## Independent full-model oracle

`hf_model.py` saves a tiny Qwen3.5 MoE checkpoint and loads it natively with
`from_pretrained(dtype=torch.bfloat16, attn_implementation="eager",
experts_implementation="eager")`. It runs the untouched native HF model with
one GDN + one gated-GQA layer, shared and routed FFNs in both layers.
`hf_model.json` contains original BF16 parameter tensors, per-layer residuals,
router IDs/F32 weights, logits, BF16 KV/history and F32 SSM for prefill, cached
decode, continuation, divergent fork and whole prefill.

**Explicit SSM policy:** this installed HF version lazily allocates SSM using
conv dtype by default. Before native forward, the generator preallocates F32
SSM tensors on `DynamicCache` and marks only those tensors initialized. Conv
history remains native BF16. Native cache updates/forward/recurrence are not
patched or replaced. The oracle is therefore HF eager with the requested
BF16-history/F32-SSM storage policy, not a claim about HF's default cache dtype.
Eight independent identical CPU batches avoid the documented BF16 rsqrt scalar
tail; source/package metadata is captured. No blanket `model.to(BF16)` rounds
RoPE frequencies.

The fixture exports native packed HF expert parameters into individual expert
bindings without changing values, for the model's existing per-expert schema.
This is not a claim that the strict 35B numeric checkpoint binder accepts native
packed BF16 expert tensors.

## Tests and commands

CPU-only fixture generation:

```sh
CUDA_VISIBLE_DEVICES= ATEN_CPU_CAPABILITY=avx512 /opt/conda310/bin/python crates/ferrule-model/tests/hybrid_bf16/hf_model.py
CUDA_VISIBLE_DEVICES= cargo test -p ferrule-model --no-default-features --test hybrid_bf16
```

`admission.rs` tests complete native binding/preparation, F32 storage rejection,
dense-vs-hybrid admission, default precision and (under CUDA feature without a
context) exact typed weight/KV/state budgets and external/partial-image rejection.
`contracts.rs` retains the original independent gate/combine/router witnesses.

`cuda.rs` contains this runnable **ignored** numerical acceptance test:

```text
cuda::native_bf16_hybrid_full_state_decode_cow_abort_restore_matches_hf
```

It builds the public model runner, checks all residuals/routes/logits, physically
typed KV snapshots after preempt/restore, BF16 history/F32 SSM, cached decode,
odd-frontier COW, aborted COW, divergent fork, reset and state-budget release.
BF16 boundaries use the existing dense-oracle envelope
`0.00025 + 0.012*abs(expected)`; route IDs are exact, physical BF16 words/dtypes
and rollback are exact. GPU execution has **not** been performed and no GPU
numerical acceptance is claimed.

Hidden-device CUDA12.4/SM86 compilation (the sourced script exposes GPU0, so
explicitly override on every command):

```sh
. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
CUDA_VISIBLE_DEVICES= cargo test -p ferrule-model --features cuda --no-default-features --test hybrid_bf16 --no-run
# Safe CPU contracts only; ignored GPU test does not execute:
CUDA_VISIBLE_DEVICES= cargo test -p ferrule-model --features cuda --no-default-features --test hybrid_bf16
```

Only a separately assigned GPU owner may run the ignored test. No GPU execution,
probe or ownership acquisition is part of this work. Numeric-FP8 35B, distributed
BF16 experts and full serving/CLI admission remain future integration stages,
not claims of this executable native model slice.
