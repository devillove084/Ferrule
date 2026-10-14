#!/usr/bin/env python3
"""Regenerate the tiny independent HF eager BF16 dense-GQA full-state oracle.

CPU only, no model download. The fixture records exact BF16 weights, every layer
output, logits, and every K/V element after prefill, decode, and a cache fork.
"""
import argparse
import copy
import json
import tempfile
import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = dict(architectures=["Qwen3ForCausalLM"], model_type="qwen3", torch_dtype="bfloat16",
                  hidden_act="silu", vocab_size=16, hidden_size=16, num_hidden_layers=2,
                  num_attention_heads=2, num_key_value_heads=1, head_dim=8, intermediate_size=24,
                  max_position_embeddings=16, rms_norm_eps=0.00001, rope_theta=10000.0,
                  tie_word_embeddings=False, attention_bias=False, use_sliding_window=False,
                  attention_dropout=0.0, use_cache=True, max_window_layers=2,
                  initializer_range=0.02, bos_token_id=1, eos_token_id=2)
    hf = Qwen3Config(**config)
    hf._attn_implementation = "eager"
    model = Qwen3ForCausalLM(hf).to(device="cpu", dtype=torch.bfloat16).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            seed = sum((i + 1) * ord(c) for i, c in enumerate(name)) % 251
            indices = torch.arange(p.numel(), dtype=torch.float32)
            values = (0.91 + (indices % 7) * 0.031) if p.ndim == 1 else torch.sin(indices * 0.731 + seed) * 0.18
            p.copy_(values.reshape(p.shape))
    # Use the native BF16 checkpoint-loading path. A blanket model.to(BF16)
    # also casts nonpersistent RoPE inverse-frequency buffers, changing the
    # frequencies *before* F32 trigonometry. That is not our BF16-table contract.
    original_weights = {name: p.detach().clone() for name, p in model.state_dict().items()}
    with tempfile.TemporaryDirectory(prefix="ferrule-bf16-hf-") as checkpoint:
        model.save_pretrained(checkpoint, safe_serialization=True)
        model = Qwen3ForCausalLM.from_pretrained(
            checkpoint, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True,
        ).eval()
    assert all(p.device.type == "cpu" and p.dtype == torch.bfloat16 for p in model.parameters())
    assert all(torch.equal(p, original_weights[name]) for name, p in model.state_dict().items())
    rotary = model.model.rotary_emb
    assert rotary.inv_freq.dtype == rotary.original_inv_freq.dtype == torch.float32
    flatten = lambda t: t.detach().float().reshape(-1).tolist()
    rotary_contract = dict(initialization="from_pretrained", inverse_frequency_dtype="float32",
                           table_dtype="bfloat16", inverse_frequency=flatten(rotary.inv_freq))
    weights = [{"name": name, "shape": list(p.shape), "values": flatten(p)} for name, p in model.state_dict().items()]
    captured = []
    def hook(_module, _input, output):
        captured.append(flatten(output[0] if isinstance(output, tuple) else output))
    handles = [layer.register_forward_hook(hook) for layer in model.model.layers]
    def call(tokens, cache=None):
        captured.clear()
        with torch.no_grad():
            out = model(torch.tensor([tokens]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        result = {"tokens": tokens, "logits": flatten(out.logits), "hidden": list(captured),
                  "key": [flatten(layer.keys.transpose(1, 2)) for layer in cache.layers],
                  "value": [flatten(layer.values.transpose(1, 2)) for layer in cache.layers]}
        return result, cache
    prefill, cache = call([1, 2, 3])
    fork = copy.deepcopy(cache)
    decode, cache = call([4], cache)
    branch, fork = call([7], fork)
    next_decode, cache = call([5], cache)
    for h in handles:
        h.remove()
    # Small, isolated tensors test semantic cast locations without CUDA or the
    # legacy CPU forward implementation sharing the implementation under test.
    x = torch.tensor([[0.73, -0.62, 1.19, -0.37]], dtype=torch.bfloat16)
    w = torch.tensor([1.071, 0.91, 1.03, 0.87], dtype=torch.bfloat16)
    up = torch.tensor([[0.31, 0.83, -0.45, 1.17]], dtype=torch.bfloat16)
    c = torch.tensor([[0.91, 0.87, 0.91, 0.87]], dtype=torch.bfloat16)
    s = torch.tensor([[0.42, 0.49, 0.42, 0.49]], dtype=torch.bfloat16)
    norm = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)).bfloat16() * w
    rotated = torch.cat((-x[..., 2:], x[..., :2]), dim=-1)
    boundaries = dict(input=flatten(x), weight=flatten(w), up=flatten(up), cosine=flatten(c), sine=flatten(s),
                      norm=flatten(norm), rope=flatten(x * c + rotated * s),
                      swiglu=flatten(torch.nn.functional.silu(x) * up), residual=flatten(x + up))
    result = dict(schema="ferrule.standard-bf16-hf-eager.v2", torch=torch.__version__,
                  transformers=transformers.__version__, device="cpu", dtype="bfloat16",
                  rotary_contract=rotary_contract, config=config, weights=weights, cases=[prefill, decode, branch, next_decode], boundaries=boundaries)
    with open(args.output, "w") as f:
        json.dump(result, f, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
