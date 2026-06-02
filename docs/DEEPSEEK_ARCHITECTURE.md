# DeepSeek-V2/V3 Architecture in Ferrule

This document describes the DeepSeek model architecture implementation in `crates/ferrule-candle/src/family/deepseek.rs`.

## Architecture Overview

DeepSeek-V2/V3 introduces two key innovations over standard transformers.

### 1. MLA (Multi-head Latent Attention)

**Problem:** Standard MHA stores full K and V in the KV cache, consuming `2 × n_layers × seq_len × n_heads × d_head` elements.

**Solution:** MLA compresses K/V into a low-rank latent space `d_c` (typically 512), then up-projects on demand. Only `d_c + d_r` dimensions per position are cached.

```
Standard MHA:
  K = X @ W_K     [seq, d] @ [d, n_heads × d_head]

MLA:
  c_KV = X @ W_DKV     [seq, d] @ [d, d_c + d_r]    ← compress
  K    = c_KV @ W_UK   [seq, d_c] @ [d_c, n_heads × d_q]  ← up-project
  V    = c_KV @ W_UV   [seq, d_c] @ [d_c, n_heads × d_v]
```

**Memory savings:** For a 5120-dim model with 128 heads and `d_c=512`, `d_r=64`:
- Standard KV: `2 × 128 × (128 + 64)` = 49,152 floats/position
- MLA KV: `512 + 64` = 576 floats/position → **~1.2%**

**Decoupled RoPE:** K is split into content (`K_nope`, no RoPE) and positional (`K_rope`, with RoPE) parts. This allows the content cache to be position-agnostic.

### 2. MoE (Mixture of Experts)

**Problem:** Dense FFN applies the same computation to all tokens, wasting capacity.

**Solution:** Route each token to top-K experts from a pool of N, via a learned gating network. Shared experts are always active.

```
output = x + shared_experts(norm(x)) + Σ routed_expert_i(norm(x)) × weight_i

Router: x @ W_router + bias → top-k → softmax → weights
Expert: SwiGLU(x) = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down
```

**Auxiliary-Loss-Free Load Balancing (DeepSeek-V3):**
Each expert maintains a dynamic bias. Selected experts decrease bias, unselected increase. This naturally balances utilization without gradient penalties.

## Module Hierarchy

```
DeepSeekModel                          (full model)
├── Embedding                         (vocab_size → d_model)
├── [DeepSeekBlock × n_layers]        (transformer layers)
│   ├── RMSNorm → MLA → Residual
│   │   └── MlaAttention              (compressed attention with KV cache)
│   └── RMSNorm → MoE → Residual
│       ├── Shared Experts            (always active)
│       ├── Router                    (Top-K gating)
│       └── Routed Experts            (sparse activation)
├── RMSNorm                            (final norm)
└── lm_head                            (d_model → vocab_size)
```

## KV Cache

MLA's incremental decoding uses cached `c_kv = [c_kv_main | k_rope_raw]`:

```
Prefill:
  input [1, P, d] → forward → cache [1, P, d_c + d_r]

Decode step i:
  input [1, 1, d]  (single new token)
  → compute new c_kv [1, 1, d_c + d_r]
  → concat with cache → [1, P+i, d_c + d_r]
  → Q: 1 position, K/V: P+i positions
  → attention → output token i
```

## Weight Loading

Weights are loaded from HuggingFace-format safetensors via VarBuilder:

```
Model directory:
  config.json           → DeepSeekV2Config (serde deserialize)
  *.safetensors         → VarBuilder::from_mmaped_safetensors()
  tokenizer.json        → FerruleTokenizer (tokenizers crate)
```

**Weight key mapping:**

| Component | HF Key |
|-----------|--------|
| Embedding | model.embed_tokens |
| Layer N attn norm | model.layers.{N}.input_layernorm |
| Q comp/decomp | model.layers.{N}.self_attn.q_a_proj, q_b_proj |
| KV comp/decomp | model.layers.{N}.self_attn.kv_a_proj_with_mqa, kv_b_proj |
| Output proj | model.layers.{N}.self_attn.o_proj |
| Layer N ffn norm | model.layers.{N}.post_attention_layernorm |
| Router | model.layers.{N}.mlp.gate.router |
| Expert J | model.layers.{N}.mlp.experts.{J}.{gate,up,down}_proj |
| Shared J | model.layers.{N}.mlp.shared_experts.{J}.{gate,up,down}_proj |
| Final norm | model.norm |
| LM head | lm_head |

## Integration with Ferrule

DeepSeek is registered as a first-class model family alongside Llama:

- `RealBackend::DeepSeek(Arc<DeepSeekModel>)` — model storage
- `CandleSession::DeepSeek(DeepSeekPolicySession)` — inference session with KV cache
- `PolicyModel` trait implementation — standard Ferrule agent interface

Usage:
```toml
[model]
backend = "real"
family = "deepseek"
model_id = "./models/deepseek-v2-lite"
device = "cuda:0"
```

## Limitations

1. **Per-token MoE forward:** Each token is forwarded through experts individually. Production implementations batch tokens per expert for efficiency.
2. **CPU Top-K:** Router uses CPU sorting. GPU-accelerated top-k would be faster.
3. **No Flash Attention:** Standard scaled dot-product attention is used.
4. **No quantization:** Only FP32/FP16 supported. Model must fit in VRAM.

These are acceptable for a learning framework (Ferrule's goal is minimal proving-ground, not production serving). For production inference, use SGLang/vLLM.

## Files Changed

| File | Change |
|------|--------|
| `crates/ferrule-candle/src/family/deepseek.rs` | New: ~1000 lines, full DeepSeek architecture |
| `crates/ferrule-candle/src/family/mod.rs` | Add `pub mod deepseek;` |
| `crates/ferrule-candle/src/lib.rs` | Register DeepSeek family: RealBackend, CandleSession, PolicyModel |
