"""Capture native HF same-input tensors, CPU only; never replace HF arithmetic.

Reuse the model handoff's tiny checkpoint loader and passive TorchFunctionMode
observer, not its arithmetic comparisons. Expected tensors are returned by native
HF attention/shared gate/experts. No Ferrule import, copied expert forward or GPU.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['ATEN_CPU_CAPABILITY'] = 'avx512'
ROOT = Path(__file__).resolve().parents[4]
HELPER = ROOT / 'crates/ferrule-model/tests/hybrid_bf16/hf_boundaries.py'
spec = importlib.util.spec_from_file_location('native_boundary_capture', HELPER)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
torch = native.torch


def capture():
    fields = {}

    def save(name, value):
        fields[name] = native.values(value) if isinstance(value, torch.Tensor) else value

    def gate(name, observer):
        assert len(observer.gates) == 1
        x, g, sigmoid, result = observer.gates[0]
        assert all(t.dtype == torch.bfloat16 for t in (x, g, sigmoid, result))
        save(name + '_shape', [x.numel() // x.shape[-1], x.shape[-1]])
        for key, t in [('input', x), ('gate', g), ('sigmoid', sigmoid), ('expected', result)]:
            save(name + '_' + key, t)

    with torch.no_grad():
        model = native.native_model()
        x = torch.linspace(-1, 1, 128).reshape(8, 16).bfloat16()
        mlp = model.model.layers[0].mlp
        with native.Capture() as routed:
            probabilities, weights, indices = mlp.gate(x)
            expected = mlp.experts(x, indices, weights)
        assert routed.linears[0].dtype == torch.bfloat16
        assert probabilities.dtype == weights.dtype == torch.float32
        save('router_logits', routed.linears[0])
        save('router_probabilities', probabilities)
        save('router_weights', weights)
        values, rows, route_weights = [], [], []
        # Only pack captured down-projection rows and native route metadata.
        # Expected combine is the native experts result, not this packing code.
        hits = sorted(set(indices.flatten().tolist()))
        assert len(routed.linears) == 1 + 2 * len(hits)
        for ordinal, expert in enumerate(hits):
            positions, tokens = torch.where(indices.T == expert)
            down = routed.linears[2 + 2 * ordinal]
            assert down.dtype == torch.bfloat16
            values.append(down)
            rows.append(tokens)
            route_weights.append(weights[tokens, positions])
        save('combine_shape', [8, 16])
        save('combine_input', torch.cat(values))
        save('combine_rows', torch.cat(rows))
        save('combine_weights', torch.cat(route_weights))
        save('combine_expected', expected)
        with native.Capture() as shared:
            mlp(x.reshape(8, 1, 16))
        gate('shared', shared)
        hidden = x.reshape(8, 1, 16)
        rotary = model.model.rotary_emb(hidden, torch.arange(8).reshape(8, 1))
        with native.Capture() as attention:
            model.model.layers[1].self_attn(hidden, rotary, attention_mask=None)
        gate('attention', attention)
        g = torch.tensor([-0.9375], dtype=torch.bfloat16)
        scalar = torch.tensor([0.515625], dtype=torch.bfloat16)
        save('scalar_expected', scalar * g.sigmoid())
    text = ''.join(k + ' ' + ' '.join(str(v) for v in values) + '\n' for k, values in fields.items())
    provenance = {
        'device': 'cpu', 'torch': torch.__version__, 'transformers': native.transformers.__version__,
        'hf_source_sha256': hashlib.sha256(Path(inspect.getfile(native.hf)).read_bytes()).hexdigest(),
        'capture_helper_sha256': hashlib.sha256(HELPER.read_bytes()).hexdigest(),
        'fixture_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'seed': 35016, 'loading': 'from_pretrained(dtype=torch.bfloat16)',
        'experts': 'native eager, ascending expert order', 'attention': 'native eager',
        'projection_dtype': 'bfloat16', 'router_logits_dtype': 'bfloat16',
        'route_weights_dtype': 'float32', 'output_dtype': 'bfloat16',
        'scope': 'operator boundary captures only, no full-model/GPU acceptance',
    }
    return text, json.dumps(provenance, indent=2) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    text, provenance = capture()
    for suffix, content in [('.txt', text), ('.json', provenance)]:
        path = Path(__file__).with_suffix(suffix)
        if args.write:
            path.write_text(content)
        else:
            assert path.read_text() == content, path
    print('Native HF CPU capture/replay passed; GPU was not used')
