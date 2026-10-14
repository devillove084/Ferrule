# Dense BF16 independent HF oracle

Regenerate with `CUDA_VISIBLE_DEVICES= python scripts/standard_bf16_reference.py
crates/ferrule-model/tests/fixtures/standard_bf16_hf.json` using the recorded
PyTorch/Transformers versions. No model download, GPU, vendor patch, or Ferrule
operator replay is involved.

Schema v2 uses Hugging Face's native `from_pretrained(dtype=torch.bfloat16,
attn_implementation="eager", local_files_only=True)` on a temporary synthetic
checkpoint. Every parameter remains BF16; both nonpersistent RoPE inverse
frequency buffers must remain F32. Native HF evaluates angle/trigonometry in
F32 and then casts the resulting cos/sin tables to BF16. Loading is asserted to
preserve every checkpoint weight exactly. The fixture records the initialization
path, inverse-frequency dtype and exact values, and table dtype.

## Why v1 was invalid for this contract

The original generator instantiated HF and called `model.to(bfloat16)`. That
also rounded the nonpersistent inverse-frequency buffer. For head dimension 8
and theta 10000, `0.1` became `0.10009765625`. At position 2, BF16 sine became
`0.19921875` rather than `0.1982421875`. This is a different RoPE configuration,
not BF16 tensor-boundary arithmetic.

CUDA12.4/SM86 per-operation captures established that all pre-RoPE norm and
projection tensors were identical. With the very same Q/K inputs, native HF
checkpoint loading matched CUDA RoPE exactly, while v1 differed. The discrepancy
propagated through second-layer attention to hidden element 33:
`-0.046875` versus the old oracle's `-0.05419921875`. Native checkpoint loading
produced `-0.046875`, matching CUDA. All prefill hidden tensors and logits were
identical with the corrected loading path, without a production arithmetic fix.

The weights, model config, isolated boundary tensors and tolerance remain
unchanged. `hf_checkpoint_loading_preserves_f32_rope_frequencies` pins this
regression. `FERRULE_BF16_TRACE_DIR` enables full per-operator capture in the
ignored CUDA test; `BF16_ORACLE` log records report maxima for every comparison.
The full test checks all layers' hidden state, all logits, all K/V words across
prefill/decode/forked-COW/continuation, preemption/restore and rollback, rather
than just top-token agreement. It retains the original strict tolerance
`0.00025 + 0.012 * abs(reference)` and physical BF16 assertions.

This dense subset does not certify GDN/MoE, full Qwen3.5 BF16 or W8 fairness.
