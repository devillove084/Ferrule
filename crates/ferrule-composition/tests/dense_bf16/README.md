# Native dense Qwen3 BF16 serving profile

Public selection:

```rust,ignore
ServingModelOptions {
    model_precision: Some(ModelPrecision::Bf16),
    ..Default::default()
}
```

Pass `BackendSelection::Cuda` to `ResidentModelPlanner::prepare_serving`.
`build()` returns the existing object-safe session engine. It uses the existing
single-stage `PipelineInferenceEngine` / standard CUDA segment / transaction
journal, not the F32 hybrid decoder or an additional KV manager.

CLI (a native BF16 **dense Qwen3** checkpoint, not Qwen3.5 or a quantized export):

```sh
ferrule serve /path/to/Qwen3 --backend cuda --model-precision bf16 \
  --engine auto --devices 0 --max-tensor-mb 2048 --kv-cache-mb 1024
```

The tensor limit bounds each original checkpoint parameter, not the entire
model. Set it high enough for the embedding/output tensors. No conversion or
CPU/F32 storage fallback is offered. Raw safetensors dtype/shape, source identity,
query/key norms, ordinary RMSNorm, unclamped SwiGLU, bias-free K8 projections and
split-half unscaled prefix RoPE are validated by the existing strict binder and
`Bf16DenseCudaProfile`. Inverse frequencies and trigonometry are F32; cosine/sine
tables and semantic RoPE boundaries are BF16, matching native HF loading rather
than blanket `model.to(bfloat16)` buffer conversion.

Weights (deduplicating tied identities), activation/residual Rows and K/V planes
are resident BF16. GEMM uses BF16 operands and F32 accumulation/output scratch,
then explicitly rounds to BF16. Q expansion and attention output scratch are
F32, not F32 KV or persistent activation mirrors. Normalization, attention
score/probability, SiLU/product and residual rounding retain the previously
validated semantic boundaries. The LM head produces BF16 device logits; the
existing serving ABI receives their exact expansion to F32. This is separate
from numeric FP8/W8A16, whose model/KV boundaries remain F32.

`dense_bf16_memory_plan()` reports native device weight bytes, rotary table upper
bound, physical typed-KV bytes (including transaction shadows), BF16 Rows, F32
work buffers, metadata and allocator headroom. No CUDA context is created by
planning. The owner checks the actual provider's zero additional GEMM workspace
contract and free device memory before weight upload. Headroom is 64 MiB + 10%
of the payload/work bound, not a reservation or protection against other GPU
processes. The existing materializer retains host checkpoint payloads and may
retain decoded host parameter representations; this plan is **not** host-RSS
admission and does not claim every host allocation is BF16. Transport uses the
existing bounded 64-KiB pinned slots. No F32 device weight image is retained.

Supported topology is strictly PP/TP/EP/DP/SP/CP=1, one CUDA device and thread
owners; no process/restart, partial depth, expert policies or prefix cache.
HostKV decode and IdleReclaim are F32 policies and rejected with BF16. Normal
retained-session turns use the same existing session/journal API. Scheduling is
serial with mixed batches/cohort deferral/proposals disabled; effective changes
appear in the resolution report. `--engine resident` is rejected rather than
pretending this standard pipeline is the hybrid resident runner.

Omitting model precision preserves all existing profiles. Explicit F32 checks
that the selected profile actually has F32 model boundaries (legacy CPU BF16
compatibility requires omission). It does not rename or disable numeric W8A16.

## Tests

- `cargo test -p ferrule-composition --test dense_bf16`: metadata admission, native
  dtype, exact/tied weight budgets, BF16 vs F32 KV bytes, bounded Rows/work and
  fail-closed policies. No GPU needed.
- CLI tests: `cargo test -p ferrule-cli --bin ferrule model_precision`.
- `reference.py`: CPU-only native HF loading of the independent two-layer
  fixture; writes `hf_serving.json`. No download or Ferrule forward code.
- Ignored `public_factory_bf16_greedy_decode_and_retained_turns_match_native_hf`:
  creates the actual public CUDA factory, tests prefill chunks 1/3, greedy
  decode, retained continuation, reset and shutdown; compares 12 selected tokens
  and logits to independent HF. Full logits/per-layer KV/COW/restore/rollback
  remain covered by the existing model `standard_bf16_cuda` full-state test;
  the public serving ABI exposes selected-token logits, not full tensors.

The new public-factory GPU test has only been **compiled**, not executed in this
change. GPU ownership was already released. All CUDA compilation uses CUDA 12.4
and SM86 with `CUDA_VISIBLE_DEVICES=` after sourcing the toolchain environment.
There is no Qwen3.5/GDN/MoE/35B support or performance/fairness claim. A future
same-GPU comparison must fix checkpoint, tokenizer, corpus, context and decode
policy in both Ferrule and vLLM and validate outputs before timing.
