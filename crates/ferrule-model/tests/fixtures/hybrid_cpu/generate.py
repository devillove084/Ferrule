"""Regenerate the generic hybrid F32 oracle with Transformers 5.2.0 CPU eager.

Uses the installed model implementation, not a second Ferrule forward.
Run: /opt/conda310/bin/python crates/ferrule-model/tests/fixtures/hybrid_cpu/generate.py
"""
import json
from pathlib import Path
import torch
import transformers
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

torch.set_num_threads(1)
config = Qwen3_5TextConfig(
    vocab_size=11, hidden_size=8, intermediate_size=12,
    num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
    head_dim=8, linear_conv_kernel_dim=3, linear_key_head_dim=3,
    linear_value_head_dim=2, linear_num_key_heads=1, linear_num_value_heads=2,
    layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                     "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0]},
    rms_norm_eps=1e-6, tie_word_embeddings=False, dtype="float32",
    attn_implementation="eager",
)
model = Qwen3_5ForCausalLM(config).float().eval()
with torch.no_grad():
    for number, (name, parameter) in enumerate(model.named_parameters()):
        x = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
        values = torch.sin(x * 0.17 + number * 0.31) * 0.2
        if name.endswith("linear_attn.norm.weight"):
            values = 0.8 + values
        if name.endswith("A_log"):
            values = -0.5 + values
        parameter.copy_(values)

def canonical(name):
    if name == "model.embed_tokens.weight": return "token_embedding.weight"
    if name == "model.norm.weight": return "final_norm.weight"
    if name == "lm_head.weight": return "output.weight"
    prefix, layer, suffix = name.split(".", 3)[1:]
    assert prefix == "layers"
    suffix = suffix.replace("input_layernorm", "input_norm").replace("post_attention_layernorm", "post_attention_norm")
    renames = {"self_attn.q_proj": "attention.query", "self_attn.k_proj": "attention.key",
        "self_attn.v_proj": "attention.value", "self_attn.o_proj": "attention.output",
        "self_attn.q_norm": "attention.query_norm", "self_attn.k_norm": "attention.key_norm",
        "linear_attn.in_proj_qkv": "attention.qkv", "linear_attn.in_proj_z": "attention.z",
        "linear_attn.in_proj_b": "attention.beta", "linear_attn.in_proj_a": "attention.a",
        "linear_attn.conv1d": "attention.conv", "linear_attn.norm": "attention.norm",
        "linear_attn.out_proj": "attention.output", "linear_attn.A_log": "attention.a_log.weight",
        "linear_attn.dt_bias": "attention.dt_bias.weight", "mlp.gate_proj": "feed_forward.gate",
        "mlp.up_proj": "feed_forward.up", "mlp.down_proj": "feed_forward.down"}
    for old, new in renames.items():
        if suffix == old or suffix.startswith(old + "."):
            suffix = new + suffix[len(old):]
            break
    return f"layers.{layer}.{suffix}"

def cache_json(cache):
    return {str(i): {"conv": cache.conv_states[i].flatten().tolist(),
                     "recurrent": cache.recurrent_states[i].flatten().tolist()}
            for i, kind in enumerate(config.layer_types) if kind == "linear_attention"}

tokens = [1, 3, 2, 5, 7]
with torch.no_grad():
    full = model(torch.tensor([tokens]), use_cache=True)
    prefill = model(torch.tensor([tokens[:3]]), use_cache=True)
    prefill_states = cache_json(prefill.past_key_values)
    fourth = model(torch.tensor([[tokens[3]]]), past_key_values=prefill.past_key_values, use_cache=True)
    fifth = model(torch.tensor([[tokens[4]]]), past_key_values=fourth.past_key_values, use_cache=True)
    torch.testing.assert_close(fifth.logits[0, 0], full.logits[0, -1], rtol=1e-4, atol=1e-5)
fixture = {"transformers": transformers.__version__, "tokens": tokens,
    "tensors": {canonical(n): {"shape": list(p.shape), "values": p.detach().flatten().tolist()}
                for n, p in model.named_parameters()},
    "logits": full.logits[0].tolist(), "states": cache_json(full.past_key_values),
    "prefill_states": prefill_states, "decoded_states": cache_json(fifth.past_key_values)}
Path(__file__).with_name("oracle.json").write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
print("Wrote", len(fixture["tensors"]), "tensors; TF", transformers.__version__)
