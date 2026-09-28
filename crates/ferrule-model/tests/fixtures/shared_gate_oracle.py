"""Regenerate a tiny F32 oracle using local Transformers (no pretrained weights).

Formula verified against local vLLM qwen3_5.Qwen3_5DecoderLayer ->
qwen3_next.Qwen3NextSparseMoeBlock (renormalize=True) and
qwen2_moe.Qwen2MoeMLP.forward (sigmoid(expert_gate(x)) * out).
Default Rust tests consume the checked-in JSON; Python is not a test dependency.
"""
import json
from pathlib import Path
import torch
import transformers
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

torch.set_num_threads(1)
c = Qwen3_5MoeTextConfig(hidden_size=4, num_experts=3, num_experts_per_tok=2,
                         moe_intermediate_size=3, shared_expert_intermediate_size=5)
m = Qwen3_5MoeSparseMoeBlock(c).float().eval()
with torch.no_grad():
    for index, (_, p) in enumerate(m.named_parameters()):
        p.copy_((((torch.arange(p.numel()).reshape(p.shape) * 7 + index * 11) % 37) - 18).float() / 24)
x = torch.tensor([[1.0, -0.5, 0.25, 2.0], [-2., 1.0, 0.5, -0.25],
                  [0., 0., 0., 0.], [0.2, -1.2, 2.5, 0.1]])
weights = {}
def add(name, value):
    weights[name] = {"shape": list(value.shape), "values": value.detach().flatten().tolist()}
add("router", m.gate.weight)
add("shared_gate", m.shared_expert_gate.weight)
for projection in ("gate", "up", "down"):
    add("shared." + projection, getattr(m.shared_expert, projection + "_proj").weight)
for e in range(c.num_experts):
    gate, up = m.experts.gate_up_proj[e].chunk(2, dim=0)
    add(f"expert.{e}.gate", gate)
    add(f"expert.{e}.up", up)
    add(f"expert.{e}.down", m.experts.down_proj[e])
with torch.no_grad():
    _, scores, selected = m.gate(x)
    routed = m.experts(x, selected, scores)
    shared = m.shared_expert(x)
    out = m(x.unsqueeze(0)).squeeze(0)
    expected = routed + torch.sigmoid(m.shared_expert_gate(x)) * shared
    torch.testing.assert_close(out, expected)
    data = {"transformers": transformers.__version__, "hidden": 4, "expert_intermediate": 3,
            "shared_intermediate": 5, "num_experts": 3, "top_k": 2, "weights": weights,
            "input": x.tolist(), "output": out.tolist(), "routed": routed.tolist(),
            "ungated": (routed + shared).tolist(), "zero_gate": (routed + 0.5 * shared).tolist()}
Path(__file__).with_suffix(".json").write_text(json.dumps(data, indent=2) + "\n")
print("Generated shared gate oracle, Transformers", transformers.__version__)
