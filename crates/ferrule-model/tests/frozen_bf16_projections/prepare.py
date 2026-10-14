#!/usr/bin/env python3
"""CPU-only frozen projection extraction; never computes an expected output."""
import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location(
    'qkv_extract', ROOT / 'crates/ferrule-backend/tests/numeric_bf16_tiled/extract_qkv.py')
util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(util)
CAUSAL = Path('target/validation/bf16-full40-causal-20261004')
TRACE = Path('target/validation/bf16-sliced-model-20261005/first-two-layers')
HF = CAUSAL / 'hf-first-two-layers.safetensors'
PROVENANCE = CAUSAL / 'torch-probe-source-payloads.json'
PROFILE = Path('target/validation/numeric-bf16-gate-20261004-v2/profile-reference/manifest.json')
PINS = {
    str(TRACE / 'events.jsonl'): '4329ea88bc3951bc252259722ab32ee282eceddeb4e4a54e0cef2c9710d30d35',
    str(HF): '41745fc2edaa2616e0018c7311144ea19bd6106fd5bb66835a7c0b800664d071',
    str(PROVENANCE): '13e7722abf398776ca1fd960d2ffa08dd88642a8d8ba2cba557032f4a0d7f406',
    str(PROFILE): '351495921233e0158eb2a4507c415d173c895a6086d5905b7a72d82de71849f9',
}
# role, independent module hook, N, K
SPECS = {
    'linear_attention.query_key_value': ('LinearAttentionQkv', 'linear_attn.in_proj_qkv', 8192, 2048),
    'linear_attention.gate': ('LinearAttentionZ', 'linear_attn.in_proj_z', 4096, 2048),
    'linear_attention.output': ('AttentionOutput', 'linear_attn.out_proj', 2048, 4096),
    'feed_forward.shared_expert.gate': ('SharedExpertGate', 'mlp.shared_expert.gate_proj', 512, 2048),
    'feed_forward.shared_expert.up': ('SharedExpertUp', 'mlp.shared_expert.up_proj', 512, 2048),
    'feed_forward.shared_expert.down': ('SharedExpertDown', 'mlp.shared_expert.down_proj', 2048, 512),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_pins(root, pins=PINS):
    for name, expected in pins.items():
        require(util.sha256(root / name) == expected, 'source SHA256 mismatch: ' + name)


def identify(event):
    match = re.fullmatch(r'layers\.([01])\.(.+)\.weight', event['parameter'])
    require(match is not None, 'unknown numeric parameter')
    layer, suffix = int(match[1]), match[2]
    require(event['layer'] == layer, 'layer mismatch')
    if suffix in SPECS:
        role, hook, n, k = SPECS[suffix]
    else:
        expert = re.fullmatch(r'feed_forward\.experts\.(\d+)\.(gate|up|down)', suffix)
        require(expert is not None and int(expert[1]) < 256, 'unknown numeric role')
        part = expert[2]
        role, hook = 'RoutedExpert' + part.title(), None
        n, k = (2048, 512) if part == 'down' else (512, 2048)
    require(event['shape'] == [1, k], 'input shape mismatch')
    require(event['weight']['shape'] == [n, k], 'weight shape mismatch')
    require(event['weight']['dtype'] == 'F8E4M3', 'not raw FP8')
    require(event['scale']['dtype'] == 'Bf16', 'not paired BF16 scales')
    require(event['scale']['shape'] == [n // 128, k // 128], 'scale shape mismatch')
    return layer, role, hook, n, k


def pair(hf, hook, layer, n, k):
    prefix = f'{layer}.{hook}'
    shape = [1, 1] if hook.startswith('linear_attn.') else [1]
    return (util.tensor(hf, prefix + '.input', 'BF16', shape + [k])[0],
            util.tensor(hf, prefix + '.output', 'BF16', shape + [n])[0])


def payload(source, dtype, shape, provenance):
    path = Path(source['path'])
    pin = provenance[source['name']]
    require(pin['shard'] == path.name and pin['shape'] == shape, 'source provenance mismatch')
    require(pin['dtype'] == {'BF16': 'torch.bfloat16', 'F8_E4M3': 'torch.float8_e4m3fn'}[dtype],
            'source provenance dtype mismatch')
    data, offset = util.tensor(path, source['name'], dtype, shape)
    require(offset == source['offset'] and len(data) == source['bytes'], 'descriptor mismatch')
    require(hashlib.sha256(data).hexdigest() == pin['sha256'], 'independent payload SHA mismatch')
    return data


def prepare(root, output):
    check_pins(root)
    events = [json.loads(s) for s in (root / TRACE / 'events.jsonl').read_text().splitlines()]
    provenance = json.loads((root / PROVENANCE).read_text())['used_payloads']
    profile = json.loads((root / PROFILE).read_text())
    require(profile['versions'] == {'torch': '2.13.0+cu129', 'transformers': '5.5.3'}, 'versions changed')
    require(profile['precision_contract']['bf16_reduced_precision_reduction'] is False, 'reduction changed')
    require(profile['precision_contract']['tf32'] is False, 'TF32 changed')
    require(not output.exists(), 'output must be fresh')
    output.mkdir(parents=True)
    hashes, cases, inventory = {}, [], []

    def put(name, data):
        with (output / name).open('xb') as f:
            f.write(data)
        hashes[name] = hashlib.sha256(data).hexdigest()

    for event in events:
        if event['event'] != 'linear.input' or event.get('scale') is None:
            continue
        layer, role, hook, n, k = identify(event)
        item = dict(layer=layer, role=role, shape=[1, n, k], parameter=event['parameter'],
                    trace_input=event['file'], weight=event['weight'], scale=event['scale'])
        if hook is None:
            item['coverage'] = 'missing independent per-expert output; down also lacks independent input'
            inventory.append(item)
            continue
        input_data, expected = pair(root / HF, hook, layer, n, k)
        captured = util.exact_words((root / TRACE / event['file']).read_bytes())
        item['production_sameinput'] = captured == input_data
        item['trace_input_sha256'] = util.sha256(root / TRACE / event['file'])
        item['coverage'] = 'independent HF input/output, not production replay'
        identity = f'l{layer}_{role}'
        put(identity + '.input.bf16', input_data)
        put(identity + '.hf.bf16', expected)
        put(identity + '.weight.fp8', payload(event['weight'], 'F8_E4M3', [n, k], provenance))
        put(identity + '.scale.bf16', payload(event['scale'], 'BF16', [n // 128, k // 128], provenance))
        cases.append(dict(id=identity, layer=layer, role=role, n=n, k=k,
                          production_sameinput=item['production_sameinput']))
        inventory.append(item)
    require(len(inventory) == 60 and len(cases) == 12, 'frozen coverage changed')
    require(len({c['id'] for c in cases}) == 12, 'duplicate case')
    audit = dict(source_sha256=PINS, numeric_projections=inventory,
                 gaps=['48 routed projections lack independent per-projection HF outputs',
                       'Both captured layers are GDN: GQA absent', 'M>1 absent',
                       'Native BF16 A/B/router/shared scalar gate/head excluded',
                       'Native HF kernel choices for out/shared/routed not captured'],
                 expected_source='original HF hook bytes; no expected synthesis or production output import')
    put('audit.json', (json.dumps(audit, indent=2, sort_keys=True) + '\n').encode())
    manifest = dict(schema='ferrule.frozen-bf16-projections.v1',
                    oracle='independent-pinned-hf-bf16',
                    hf_sha256=PINS[str(HF)], profile_sha256=PINS[str(PROFILE)],
                    torch='2.13.0+cu129', transformers='5.5.3',
                    model_boundary='bf16', accumulation='f32',
                    allow_tf32=False, allow_bf16_reduced_precision_reduction=False,
                    numeric_count=60, missing_expert_pairs=48, cases=cases)
    put('manifest.json', (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode())
    (output / 'SHA256SUMS').write_text(''.join(f'{sha}  {name}\n' for name, sha in sorted(hashes.items())))
    print(json.dumps(dict(fixture=str(output), sha256sums=util.sha256(output / 'SHA256SUMS'),
                          independent_pairs=len(cases), missing_expert_pairs=48,
                          production_sameinput=sum(c['production_sameinput'] for c in cases))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.output)
