"""Bounded first-four-prefill-layer diagnosis from opt-in full40 test/oracle captures.

No model forward is replaced. Same-input projections are recomputed in independent
CPU Torch, with explicit FP8 scale multiply + integer BF16 RNE from the oracle.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from analyze_numeric_trace import diff, tensor

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
import qwen35_fp8_reference as ref
ref.torch = torch


def metric(a, b):
    return diff(a, b, atol=2e-4, rtol=2e-4)



def packed_vector_reduction(x, weight):
    """Diagnostic replay of the documented 32-lane/four-accumulator F32 order.

    NOT an acceptance oracle. BF16 products are exact in F32 for these finite,
    normal-range operands; F64 multiply-add then F32 conversion models each RN
    FMA independently of CUDA. No output from Ferrule is used by this replay.
    """
    assert x.shape[1] == weight.shape[1] and x.shape[1] % 128 == 0
    assert torch.equal(x, ref.independent_bf16_rne(x))
    assert torch.equal(weight, ref.independent_bf16_rne(weight))
    a, w = x.double().numpy(), weight.double().numpy()
    accumulator = np.zeros((len(a), len(w), 32, 4), dtype=np.float32)
    for start in range(0, a.shape[1], 128):
        product = a[:, None, start:start+128] * w[None, :, start:start+128]
        assert np.array_equal(product, product.astype(np.float32).astype(np.float64))
        accumulator = (accumulator.astype(np.float64) + product.reshape(accumulator.shape)).astype(np.float32)
    sums = (accumulator[..., 0] + accumulator[..., 1]) + (accumulator[..., 2] + accumulator[..., 3])
    for offset in (16, 8, 4, 2, 1):
        sums[..., :32-offset] = sums[..., :32-offset] + sums[..., offset:]
    return torch.from_numpy(sums[..., 0].copy())


def bin_effect(x, expected_x, weight, actual, expected):
    a, e = ref.independent_bf16_rne(x), ref.independent_bf16_rne(expected_x)
    indices = torch.nonzero(a != e)
    delta = F.linear((a - e).double(), weight.double()).float()
    explained = expected.reshape_as(actual).double() + delta.double()
    examples = []
    for row, col in indices[:16].tolist():
        av, ev = float(x[row, col]), float(expected_x[row, col])
        ar, er = float(a[row, col]), float(e[row, col])
        adjacent = abs(int(a[row, col].bfloat16().view(torch.int16))
                       - int(e[row, col].bfloat16().view(torch.int16))) == 1
        midpoint = (ar + er) / 2 if adjacent else None
        examples.append(dict(row=row, column=col, actual=av, reference=ev,
            actual_rne=ar, reference_rne=er, adjacent_bf16_bins=adjacent, midpoint=midpoint,
            actual_minus_midpoint=av-midpoint if adjacent else None,
            reference_minus_midpoint=ev-midpoint if adjacent else None,
            nonzero_weight_column=int(torch.count_nonzero(weight[:, col]))))
    return dict(changed_elements=len(indices), examples=examples,
                predicted_max_abs_delta=float(delta.abs().max()),
                gpu_vs_oracle_plus_operand_delta=metric(actual, explained))


def analyze(trace, oracle_path, max_layers=4):
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    events = [json.loads(line) for line in (trace / 'events.jsonl').read_text().splitlines()]
    assert 1 <= max_layers <= 4
    assert events and all(e['precision'] == 'bf16_rne' and e['stage'] == 'prefill' for e in events)
    def read(event):
        value = torch.frombuffer(bytearray((trace / event['file']).read_bytes()), dtype=torch.float32)
        assert value.numel() == event['shape'][0] * event['shape'][1]
        return value.reshape(event['shape'])
    report = dict(atol=2e-4, rtol=2e-4, boundaries=[], same_input=[], norms=[], routes=[])
    with safe_open(oracle_path, framework='pt', device='cpu') as oracle:
        inputs = {}
        residual_outputs = {}
        for event in events:
            layer, name = event['layer'], event['event']
            if layer is not None and layer >= max_layers:
                continue
            label = event['parameter'] or name
            actual = read(event)
            expected = None
            module_name = None
            if event['weight']:
                module_name = event['weight']['name'].replace('model.language_model.', 'model.').removesuffix('.weight')
            prefix = 'prefill.trace.' + module_name if module_name else None
            if name in ('linear.input', 'norm.input'):
                inputs[event['parameter']] = actual
            if name == 'embedding':
                expected = oracle.get_tensor('prefill.embedding')
            elif name.startswith(('linear.', 'norm.')):
                suffix = 'input.0' if name.endswith('input') else 'output.0'
                key = f'{prefix}.{suffix}'
                if key in oracle.keys():
                    expected = oracle.get_tensor(key)
                if event['role'] == 'RouterLogits' and name == 'linear.output':
                    expected = F.linear(oracle.get_tensor(prefix + '.input.0'), tensor(event['weight']).float())
            elif name in ('gdn.recurrent_output', 'gdn.gated'):
                suffix = 'input.0' if name == 'gdn.recurrent_output' else 'output.0'
                expected = oracle.get_tensor(f'prefill.trace.model.layers.{layer}.linear_attn.norm.{suffix}')
            elif name in ('gdn.conv_state', 'gdn.recurrent_state'):
                field = 'conv_states' if name == 'gdn.conv_state' else 'recurrent_states'
                expected = oracle.get_tensor(f'prefill.cache.{field}.{layer:02d}')
            elif name == 'gdn.convolved':
                conv = oracle.get_tensor(f'prefill.trace.model.layers.{layer}.linear_attn.conv1d.output.0')
                expected = F.silu(conv[:, :, :actual.shape[0]]).transpose(1, 2)
            elif name == 'routed.output':
                expected = oracle.get_tensor(f'prefill.trace.model.layers.{layer}.mlp.experts.output.0')
            elif name == 'residual.output':
                index = residual_outputs.get(layer, 0)
                residual_outputs[layer] = index + 1
                if index == 0:
                    hidden = oracle.get_tensor('prefill.embedding' if layer == 0 else f'prefill.layers.{layer-1:02d}.hidden')
                    expected = hidden + oracle.get_tensor(f'prefill.layers.{layer:02d}.attention')
                elif index == 1:
                    expected = oracle.get_tensor(f'prefill.layers.{layer:02d}.moe')
                elif index == 2:
                    expected = oracle.get_tensor(f'prefill.layers.{layer:02d}.hidden')
                else:
                    raise ValueError('Unexpected layer residual topology')
            if expected is not None:
                result = dict(sequence=event['sequence'], layer=layer, event=name, parameter=label,
                              **metric(actual, expected))
                report['boundaries'].append(result)
            if name == 'norm.output':
                x = inputs[event['parameter']]
                weight = tensor(event['weight']).float()
                x = x.reshape(-1, weight.numel())
                normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + event['epsilon'])
                normalized *= weight + int(event['one_plus_weight'])
                report['norms'].append(dict(layer=layer, parameter=label, same_input=metric(actual, normalized)))
            if name == 'linear.output':
                x = inputs[event['parameter']]
                raw = tensor(event['weight'])
                weight = (ref.independent_dequantize(raw, tensor(event['scale']))
                          if event['numeric'] else raw.float())
                x_operand = ref.independent_bf16_rne(x) if event['numeric'] else x
                same = F.linear(x_operand, weight)
                precise = F.linear(x_operand.double(), weight.double()).float()
                item = dict(sequence=event['sequence'], layer=layer, parameter=label,
                            numeric=event['numeric'], same_input=metric(actual, same),
                            f64_product_sum=metric(actual, precise), torch_vs_f64=metric(same, precise))
                key = prefix + '.input.0'
                if key in oracle.keys():
                    rx = oracle.get_tensor(key).reshape_as(x)
                    item['input_vs_oracle'] = metric(x, rx)
                    if event['numeric']:
                        item['rounded_input_vs_oracle'] = metric(x_operand, ref.independent_bf16_rne(rx))
                        if expected is not None:
                            item['bin_effect'] = bin_effect(x, rx, weight, actual, expected)
                if event['numeric'] and layer <= 2 and event['role'] in ('LinearAttentionQkv', 'SharedExpertGate'):
                    item['explicit_f32_reduction_replay'] = metric(actual, packed_vector_reduction(x_operand, weight))
                report['same_input'].append(item)
            if name == 'router':
                expected_ids = oracle.get_tensor(f'prefill.layers.{layer:02}.selected_expert_ids')
                ids = torch.tensor(event['ids']).reshape_as(expected_ids)
                probabilities = F.softmax(actual, dim=-1)
                scores, selected = torch.topk(probabilities, 8, dim=-1)
                scores /= scores.sum(-1, keepdim=True)
                report['routes'].append(dict(layer=layer, ids_match_oracle=bool(torch.equal(ids, expected_ids)),
                    ids_match_same_input=bool(torch.equal(ids, selected)),
                    weights_same_input=metric(torch.tensor(event['weights']), scores)))
        for key in ('first_unequal', 'first_outside', 'crossed_bf16_bins'):
            report[{'first_unequal': 'first_bitwise_difference', 'first_outside': 'first_strict_failure', 'crossed_bf16_bins': 'first_bf16_bin_difference'}[key]] = next((x for x in report['boundaries'] if
                (x[key] is not None if key in ('first_unequal', 'first_outside') else x[key] > 0)), None)
    return report



def isolate_gdn(trace, model):
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as q
    import transformers
    assert transformers.__version__ == '5.2.0'
    config = json.loads((model / 'config.json').read_text())['text_config']
    events = [json.loads(line) for line in (trace / 'events.jsonl').read_text().splitlines()]
    store = ref.TensorStore(model)
    report = []
    try:
        for layer in range(3):
            es = [event for event in events if event['layer'] == layer]
            def get(name, role=None):
                matches = [e for e in es if e['event'] == name and (role is None or e['role'] == role)]
                assert len(matches) == 1, (name, role)
                event = matches[0]
                return torch.frombuffer(bytearray((trace/event['file']).read_bytes()), dtype=torch.float32).reshape(event['shape'])
            def weight(name):
                return store.read(f'{ref.PREFIX}layers.{layer}.linear_attn.{name}').float()
            def record(label, actual, expected):
                report.append(dict(layer=layer, operation=label, **metric(actual, expected)))
            conv_w, alog, dt, norm = [weight(n) for n in ('conv1d.weight', 'A_log', 'dt_bias', 'norm.weight')]
            kh, vh = config['linear_num_key_heads'], config['linear_num_value_heads']
            kd, vd = config['linear_key_head_dim'], config['linear_value_head_dim']
            qkv = get('linear.output', 'LinearAttentionQkv')
            rows, channels = qkv.shape
            conv = F.silu(F.conv1d(F.pad(qkv.T.unsqueeze(0), (conv_w.shape[-1]-1, 0)),
                                  conv_w, groups=channels)).transpose(1, 2)
            observed_conv = get('gdn.convolved').reshape(1, rows, channels)
            record('conv+silu same QKV', observed_conv, conv)
            query, key, value = observed_conv.split([kh*kd, kh*kd, vh*vd], dim=-1)
            query = query.reshape(1, rows, kh, kd).repeat_interleave(vh//kh, dim=2)
            key = key.reshape(1, rows, kh, kd).repeat_interleave(vh//kh, dim=2)
            value = value.reshape(1, rows, vh, vd)
            a = get('linear.output', 'LinearAttentionA').reshape(1, rows, vh)
            b = get('linear.output', 'LinearAttentionBeta').reshape(1, rows, vh)
            g, beta = -alog.exp()*F.softplus(a+dt), b.sigmoid()
            output, state = q.torch_chunk_gated_delta_rule(query, key, value, g, beta,
                initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
            record('native chunk GDN same convolved/a/b, zero initial state', get('gdn.recurrent_output'), output)
            record('native chunk GDN state same convolved/a/b, zero initial state', get('gdn.recurrent_state'), state)
            recurrent = get('gdn.recurrent_output').reshape(-1, vd)
            normalized = recurrent * torch.rsqrt(recurrent.square().mean(-1, keepdim=True) + config['rms_norm_eps']) * norm
            record('GDN norm same recurrent output', get('gdn.normalized'), normalized)
            z = get('linear.output', 'LinearAttentionZ').reshape(-1, vd)
            gated = get('gdn.normalized').reshape(-1, vd) * F.silu(z)
            record('GDN SiLU gate same normalized/z', get('gdn.gated'), gated)
    finally:
        store.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, help='Also isolate native GDN on the same captured operands')
    args = parser.parse_args()
    report = analyze(args.trace, args.oracle)
    if args.model:
        report['gdn_same_input'] = isolate_gdn(args.trace, args.model)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for value in report['boundaries']:
        if value['layer'] in (None, 0, 1):
            print(value['sequence'], value['event'], value['parameter'],
                  {k:value[k] for k in ('max_abs', 'outside', 'unequal', 'crossed_bf16_bins')})
    for key in ('first_bitwise_difference', 'first_strict_failure', 'first_bf16_bin_difference'):
        print(key, report[key])
    print('Same-input failing projections:',[(x['layer'],x['parameter'],x['same_input']['outside'])
                                              for x in report['same_input'] if x['same_input']['outside']])


if __name__ == '__main__':
    main()
