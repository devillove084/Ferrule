"""CPU-only native HF boundary evidence, NOT a Ferrule hybrid model oracle.

Explicit eager attention/experts are part of this contract. No vendor patch,
replacement forward, Ferrule import, GPU context, or blanket model.to(BF16).
Run with --write to regenerate hf_boundaries.json; otherwise just verify.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ATEN_CPU_CAPABILITY"] = "avx512"

import torch
from torch.overrides import TorchFunctionMode
import transformers
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf


class Capture(TorchFunctionMode):
    """Observe native operations without substituting their implementations."""

    def __init__(self):
        self.linears = []
        self.sigmoids = {}
        self.gates = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func is torch.nn.functional.linear:
            self.linears.append(result.detach().clone())
        if func.__name__ == "sigmoid":
            self.sigmoids[result.data_ptr()] = (args[0].clone(), result.clone())
        if func.__name__ == "mul" and len(args) == 2:
            a, b = args
            sigmoid_arg = next(
                (value for value in (a, b)
                 if isinstance(value, torch.Tensor) and value.data_ptr() in self.sigmoids),
                None,
            )
            if sigmoid_arg is not None:
                gate, sigmoid = self.sigmoids[sigmoid_arg.data_ptr()]
                operand = b if sigmoid_arg is a else a
                self.gates.append((operand.clone(), gate, sigmoid, result.clone()))
        return result


def values(tensor):
    return tensor.detach().float().reshape(-1).tolist()


def native_model():
    torch.set_num_threads(1)
    torch.manual_seed(35016)
    config = hf.Qwen3_5MoeTextConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=4, linear_value_head_dim=4, linear_conv_kernel_dim=4,
        moe_intermediate_size=12, shared_expert_intermediate_size=12,
        num_experts=4, num_experts_per_tok=3, rms_norm_eps=1e-6,
    )
    with tempfile.TemporaryDirectory(prefix="ferrule-hf-hybrid-boundaries-") as directory:
        source = hf.Qwen3_5MoeForCausalLM(config).eval()
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.copy_(torch.randn_like(parameter) * 0.3)
        source.save_pretrained(directory)
        return hf.Qwen3_5MoeForCausalLM.from_pretrained(
            directory, dtype=torch.bfloat16, attn_implementation="eager",
            experts_implementation="eager",
        ).eval()


def gate_record(capture):
    assert len(capture.gates) == 1
    x, gate, sigmoid, actual = capture.gates[0]
    assert {t.dtype for t in (x, gate, sigmoid, actual)} == {torch.bfloat16}
    assert torch.equal(actual, x * gate.sigmoid())
    wrong = (x.float() * gate.float().sigmoid()).bfloat16()
    bad = (wrong != actual).reshape(-1)
    assert bad.any(), "fixture must distinguish the missing sigmoid boundary"
    index = torch.nonzero(bad).flatten()[0].item()
    broadcast_gate = gate.expand_as(x)
    broadcast_sigmoid = sigmoid.expand_as(x)
    return {
        "elements": actual.numel(), "mismatches": bad.sum().item(),
        "max_error": (wrong.float() - actual.float()).abs().max().item(),
        "x": values(x)[index], "gate": values(broadcast_gate)[index],
        "sigmoid": values(broadcast_sigmoid)[index],
        "hf": values(actual)[index], "f32_gate_then_cast": values(wrong)[index],
    }


@torch.no_grad()
def capture_boundaries():
    model = native_model()
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    assert model.config._experts_implementation == "eager"
    assert model.config._attn_implementation == "eager"
    assert model.model.rotary_emb.inv_freq.dtype == torch.float32
    assert model.model.rotary_emb.original_inv_freq.dtype == torch.float32
    x = torch.linspace(-1.0, 1.0, 128).reshape(8, 16).bfloat16()
    mlp = model.model.layers[0].mlp
    with Capture() as routed:
        probabilities, weights, indices = mlp.gate(x)
        actual = mlp.experts(x, indices, weights)
    logits = routed.linears[0]
    assert logits.dtype == torch.bfloat16
    assert probabilities.dtype == weights.dtype == torch.float32
    assert torch.equal(probabilities, logits.float().softmax(-1))
    assert actual.dtype == torch.bfloat16
    selected_only = logits.float().gather(1, indices).softmax(-1)

    # Alternative combines consume CAPTURED native down-projection outputs;
    # no expert GEMM/activation math is copied into the oracle.
    rounded = torch.zeros_like(x)
    f32_sum = torch.zeros_like(x, dtype=torch.float32)
    outputs = []
    hits = sorted(set(indices.flatten().tolist()))
    assert len(routed.linears) == 1 + 2 * len(hits)
    for ordinal, expert in enumerate(hits):
        positions, tokens = torch.where(indices.T == expert)
        down = routed.linears[2 + 2 * ordinal]
        assert down.dtype == torch.bfloat16
        route_weight = weights[tokens, positions, None]
        contribution = down * route_weight
        assert contribution.dtype == torch.float32
        rounded.index_add_(0, tokens, contribution.bfloat16())
        f32_sum.index_add_(0, tokens, contribution)
        outputs.append((expert, tokens, down, route_weight))
    assert torch.equal(rounded, actual)
    wrong = f32_sum.bfloat16()
    bad = wrong != actual
    assert bad.any()
    token, column = torch.nonzero(bad)[0].tolist()
    terms = []
    for expert, tokens, down, route_weight in outputs:
        match = torch.nonzero(tokens == token).flatten()
        if match.numel():
            row = match.item()
            terms.append({"expert": expert, "value": down[row, column].item(),
                          "weight": route_weight[row, 0].item()})

    with Capture() as shared:
        mlp(x.reshape(8, 1, 16))
    hidden = x.reshape(8, 1, 16)
    positions = torch.arange(8).reshape(8, 1)
    rotary = model.model.rotary_emb(hidden, positions)
    assert all(t.dtype == torch.bfloat16 for t in rotary)
    with Capture() as attention:
        model.model.layers[1].self_attn(hidden, rotary, attention_mask=None)

    norm = model.model.layers[0].input_layernorm
    norm_actual = norm(x)
    normalized = x.float() * (x.float().square().mean(-1, keepdim=True) + norm.eps).rsqrt()
    norm_expected = (normalized * (1.0 + norm.weight.float())).bfloat16()
    norm_wrong = (normalized.bfloat16().float() * (1.0 + norm.weight.float())).bfloat16()
    assert torch.equal(norm_actual, norm_expected)
    assert torch.any(norm_wrong != norm_actual)

    # Stable scalar witness, in addition to captures inside native modules.
    gate = torch.full((32,), -0.9375, dtype=torch.bfloat16)
    operand = torch.full_like(gate, 0.515625)
    gate_hf = (operand * gate.sigmoid())[0].item()
    gate_wrong = (operand.float() * gate.float().sigmoid()).bfloat16()[0].item()
    assert (gate_hf, gate_wrong) == (0.14453125, 0.1455078125)
    source = Path(inspect.getfile(hf)).read_bytes()
    return {
        "scope": "boundary evidence only; no Ferrule forward or GPU numerical acceptance",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "hf_source_sha256": hashlib.sha256(source).hexdigest(),
        "seed": 35016, "device": "cpu", "aten_cpu_capability": "avx512",
        "loading": "save_pretrained -> from_pretrained(dtype=torch.bfloat16)",
        "attention_implementation": "eager", "experts_implementation": "eager",
        "parameters": {n: str(p.dtype) for n, p in model.named_parameters()},
        "buffers": {n: str(b.dtype) for n, b in model.named_buffers()},
        "router": {"linear_output": str(logits.dtype),
                   "probabilities": str(probabilities.dtype), "weights": str(weights.dtype),
                   "logits": values(logits), "full_probabilities": values(probabilities),
                   "selected_weights": values(weights), "indices": indices.tolist(),
                   "selected_softmax_max_error": (weights - selected_only).abs().max().item()},
        "attention_gate": gate_record(attention), "shared_gate": gate_record(shared),
        "scalar_gate": {"x": 0.515625, "gate": -0.9375, "sigmoid": gate.sigmoid()[0].item(),
                        "hf": gate_hf, "f32_gate_then_cast": gate_wrong},
        "combine": {"elements": actual.numel(), "mismatches": bad.sum().item(),
                    "max_error": (wrong.float() - actual.float()).abs().max().item(),
                    "token": token, "column": column, "terms_expert_order": terms,
                    "hf": actual[token, column].item(), "f32_sum_then_cast": wrong[token, column].item()},
        "offset_norm": {"elements": norm_actual.numel(),
                        "early_cast_mismatches": torch.count_nonzero(norm_wrong != norm_actual).item()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    evidence = capture_boundaries()
    fixture = Path(__file__).with_suffix(".json")
    if args.write:
        fixture.write_text(json.dumps(evidence, indent=2) + "\n")
    else:
        assert evidence == json.loads(fixture.read_text()), "native HF capture or environment drift"
    print(json.dumps({k: evidence[k] for k in (
        "scope", "torch", "transformers", "hf_source_sha256", "attention_gate",
        "shared_gate", "scalar_gate", "combine", "offset_norm",
    )}, indent=2))
