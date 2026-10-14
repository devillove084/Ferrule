"""CPU-only HF native BF16 oracle; no Ferrule code is imported or copied."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Force native AVX512 BF16 vector dispatch; replicated independent batches below
# avoid the scalar CPU rsqrt tail, which double-rounds in this torch build.
os.environ["ATEN_CPU_CAPABILITY"] = "avx512"
import torch
import torch.nn.functional as F
import transformers
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.manual_seed(35016)
# Explicit HF eager baseline, not an installed optional fused implementation.
hf.FusedRMSNormGated = None
hf.causal_conv1d_fn = None
hf.causal_conv1d_update = None
hf.chunk_gated_delta_rule = None
hf.fused_recurrent_gated_delta_rule = None
config = hf.Qwen3_5TextConfig(
    vocab_size=32, hidden_size=12, intermediate_size=20, num_hidden_layers=1,
    num_attention_heads=2, num_key_value_heads=1, head_dim=6,
    layer_types=["linear_attention"], linear_num_key_heads=2,
    linear_num_value_heads=4, linear_key_head_dim=3, linear_value_head_dim=2,
    linear_conv_kernel_dim=4, rms_norm_eps=1e-6,
)
with tempfile.TemporaryDirectory(prefix="ferrule-hf-bf16-") as directory:
    source = hf.Qwen3_5ForCausalLM(config).eval()
    with torch.no_grad():
        for name, p in source.named_parameters():
            p.copy_(torch.randn_like(p) * (0.32 if "conv1d" in name else 0.41))
        source.model.layers[0].linear_attn.norm.weight.copy_(torch.tensor([0.7317, 1.1839]))
        source.model.layers[0].linear_attn.A_log.copy_(torch.tensor([-2.817, -1.329, 0.1379, 1.218]))
    source.save_pretrained(directory)
    model = hf.Qwen3_5ForCausalLM.from_pretrained(directory, dtype=torch.bfloat16).eval()
layer = model.model.layers[0].linear_attn
assert {p.dtype for p in layer.parameters()} == {torch.bfloat16}
records = []
BATCH = 8
probe = torch.full((32,), 0.00811767578125, dtype=torch.bfloat16)
assert torch.equal(probe.rsqrt(), probe.float().rsqrt().bfloat16())
def batch(x):
    return x.repeat(BATCH, *([1] * (x.ndim-1)))

def emit(name, tensor):
    if tensor.ndim >= 3 and tensor.shape[0] == BATCH:
        tensor = tensor[:1]
    values = tensor.detach().float().reshape(-1).tolist()
    records.append(name + " " + " ".join(format(v, ".9g") for v in values))

with torch.no_grad():
    for rows, nonzero, bias_enabled in [(1, True, False), (7, True, True), (65, False, False), (129, True, False)]:
        records.append(f"case {rows} 2 4 3 2 4 {int(bias_enabled)}")
        hidden = batch((torch.randn(1, rows, 12) * 0.7).bfloat16())
        raw = layer.in_proj_qkv(hidden)
        a, b = layer.in_proj_a(hidden), layer.in_proj_b(hidden)
        a[:, 0, :] = torch.tensor([-30, 25, -2.531, 0.378]).bfloat16()
        b[:, 0, :] = torch.tensor([-25, 25, -0.8137, 1.2137]).bfloat16()
        z = layer.in_proj_z(hidden).reshape(BATCH, rows, 4, 2)
        weight = layer.conv1d.weight.squeeze(1)
        history = (torch.randn(1, 20, 4) * 0.17).bfloat16() if nonzero else torch.zeros(1, 20, 4, dtype=torch.bfloat16)
        state = torch.randn(1, 4, 3, 2) * 0.11 if nonzero else torch.zeros(1, 4, 3, 2)
        history, state = batch(history), batch(state)
        bias = (torch.randn(20) * 0.13).bfloat16() if bias_enabled else None
        for name, value in [("raw", raw), ("weight", weight), ("bias", bias if bias is not None else torch.empty(0)),
                            ("history", history), ("state", state), ("a", a), ("b", b),
                            ("a_log", layer.A_log), ("dt_bias", layer.dt_bias), ("z", z), ("norm_weight", layer.norm.weight)]:
            emit(name, value)
        history_final = history.clone()
        conv = hf.torch_causal_conv1d_update(raw.transpose(1, 2), history_final, weight, bias, "silu").transpose(1, 2)
        assert conv.dtype == history_final.dtype == torch.bfloat16
        if not nonzero:
            assert torch.equal(conv, F.silu(layer.conv1d(raw.transpose(1, 2))[:, :, :rows]).transpose(1, 2))
        q, k, v = conv.split([6, 6, 8], dim=-1)
        q = q.reshape(BATCH, rows, 2, 3).repeat_interleave(2, dim=2)
        k = k.reshape(BATCH, rows, 2, 3).repeat_interleave(2, dim=2)
        v = v.reshape(BATCH, rows, 4, 2)
        beta = b.sigmoid()
        g = -layer.A_log.float().exp() * F.softplus(a.float() + layer.dt_bias)
        output, final = hf.torch_recurrent_gated_delta_rule(q, k, v, g, beta, state, True, True)
        assert output.dtype == torch.bfloat16 and final.dtype == torch.float32
        chunk, chunk_final = hf.torch_chunk_gated_delta_rule(q, k, v, g, beta, initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)
        torch.testing.assert_close(chunk.float(), output.float(), atol=2e-5, rtol=0.008)
        torch.testing.assert_close(chunk_final, final, atol=3e-6, rtol=3e-5)
        norm = layer.norm(output.reshape(-1, 2), z.reshape(-1, 2))
        for name, value in [("conv", conv), ("history_final", history_final), ("qnorm", hf.l2norm(q)),
                            ("knorm", hf.l2norm(k)), ("beta", beta), ("g", g), ("output", output),
                            ("state_final", final), ("chunk_output", chunk), ("norm", norm[:rows*4])]:
            emit(name, value)
        ht, st = history.clone(), state.clone()
        histories, states, outputs = [], [], []
        for t in range(rows):
            ct = hf.torch_causal_conv1d_update(raw[:, t:t+1].transpose(1, 2), ht, weight, bias, "silu").transpose(1, 2)
            assert torch.equal(ct, conv[:, t:t+1])
            ot, st = hf.torch_recurrent_gated_delta_rule(q[:, t:t+1], k[:, t:t+1], v[:, t:t+1], g[:, t:t+1], beta[:, t:t+1], st, True, True)
            histories.append(ht.clone()); states.append(st.clone()); outputs.append(ot)
        stepped = torch.cat(outputs, dim=1)
        assert torch.equal(stepped, output)
        assert torch.equal(st, final)
        emit("step_output", stepped)
        emit("history_steps", torch.stack(histories, dim=1))
        emit("state_steps", torch.stack(states, dim=1))
        records.append("end")
fixture = "\n".join(records) + "\n"
(HERE / "recurrent_bf16_hf.txt").write_text(fixture)
metadata = dict(
    torch=torch.__version__, transformers=transformers.__version__, seed=35016, device="cpu", aten_cpu_capability="avx512", oracle_batch_replicas=BATCH,
    cpu_rsqrt_note="8 identical independent batches avoid CPU scalar BF16 rsqrt double-rounding tails; vector BF16 rsqrt verified against F32-rsqrt then BF16",
    loading="Qwen3_5ForCausalLM.save_pretrained -> from_pretrained(dtype=torch.bfloat16)",
    parameters={n: str(p.dtype) for n, p in layer.named_parameters()},
    hf_source_sha256=hashlib.sha256(Path(inspect.getfile(hf)).read_bytes()).hexdigest(),
    fixture_sha256=hashlib.sha256(fixture.encode()).hexdigest(),
    cases=[1, 7, 65, 129], recurrent_state_dtype="torch.float32", conv_history_dtype="torch.bfloat16",
    oracle="HF torch_causal_conv1d_update, torch_recurrent_gated_delta_rule, torch_chunk_gated_delta_rule, Qwen3_5RMSNormGated",
)
(HERE / "recurrent_bf16_hf.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
