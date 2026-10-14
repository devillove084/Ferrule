#!/usr/bin/env python3
"""CPU-only extraction of pinned Hello L0 in_proj_z SAMEINPUT bytes.

Not an oracle exporter: no Torch, model execution, or expected-value synthesis.
The native HF capture is the reference; fused/tiled captures are diagnostic only.
"""
import argparse
import hashlib
import json
from pathlib import Path

from extract_qkv import exact_words, sha256, tensor

CAUSAL = Path('target/validation/bf16-full40-causal-20261004')
TILED = Path('target/validation/bf16-tiled-model-20261005/full40-first-two-layers')
HF = CAUSAL / 'hf-first-two-layers.safetensors'
PROVENANCE = CAUSAL / 'torch-probe-source-payloads.json'
PINS = {
    CAUSAL / 'first-two-layers/events.jsonl': '3379c4be7e4d9c598acdf026d30004cdd8a353d841825a0cf5497614dc7dbdad',
    CAUSAL / 'first-two-layers/0005.f32': '04c24265212cc6828ef676790aa373c1202e3f8f9198874faffb7ce03435a4fe',
    CAUSAL / 'first-two-layers/0006.f32': '80ba38f5a9e6edf6f9ab0f08c54aaf3c9a228b7f918a5734c930e84e1984f512',
    TILED / 'events.jsonl': '4329ea88bc3951bc252259722ab32ee282eceddeb4e4a54e0cef2c9710d30d35',
    TILED / '0005.f32': '04c24265212cc6828ef676790aa373c1202e3f8f9198874faffb7ce03435a4fe',
    TILED / '0006.f32': 'c4ff61bd9140abafde8444da24aef1424d6dd1be9e94ced6b2023312ef4e9479',
    HF: '41745fc2edaa2616e0018c7311144ea19bd6106fd5bb66835a7c0b800664d071',
    PROVENANCE: '13e7722abf398776ca1fd960d2ffa08dd88642a8d8ba2cba557032f4a0d7f406',
}
PAYLOAD_PINS = {
    'weight': '828b21475685542ee11d4655872ec25a5d4cf9ded3f2922949108401f1d7c4ee',
    'scale': '4ad6470feea638ff238a7691cdb7545b9bb62224e4362da344988bd97cfac474',
}


def differences(actual, expected):
    assert len(actual) == len(expected) == 8192
    return [dict(index=i // 2,
                 actual_bf16=f'0x{int.from_bytes(actual[i:i+2], "little"):04x}',
                 hf_bf16=f'0x{int.from_bytes(expected[i:i+2], "little"):04x}')
            for i in range(0, len(expected), 2) if actual[i:i+2] != expected[i:i+2]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for relative, pin in PINS.items():
        assert sha256(args.root / relative) == pin, relative
    events = [json.loads(s) for s in (args.root / TILED / 'events.jsonl').read_text().splitlines()]
    fused_events = [json.loads(s) for s in (args.root / CAUSAL / 'first-two-layers/events.jsonl').read_text().splitlines()]
    event = events[5]
    assert event == fused_events[5]
    assert event['event'] == 'linear.input' and event['layer'] == 0
    assert event['parameter'] == 'layers.0.linear_attention.gate.weight'
    assert event['shape'] == [1, 2048]
    for output in [events[6], fused_events[6]]:
        assert output['event'] == 'linear.output' and output['layer'] == 0
        assert output['parameter'] == event['parameter'] and output['shape'] == [1, 4096]
        assert output['weight'] == event['weight'] and output['scale'] == event['scale']
    provenance = json.loads((args.root / PROVENANCE).read_text())['used_payloads']
    payloads = {}
    for kind, dtype, shape, filename in [
        ('weight', 'F8_E4M3', [4096, 2048], 'weight.fp8'),
        ('scale', 'BF16', [32, 16], 'scale.bf16'),
    ]:
        source = event[kind]
        data, offset = tensor(Path(source['path']), source['name'], dtype, shape)
        assert offset == source['offset'] and len(data) == source['bytes']
        assert hashlib.sha256(data).hexdigest() == PAYLOAD_PINS[kind]
        assert provenance[source['name']]['sha256'] == PAYLOAD_PINS[kind]
        payloads[filename] = data
    hf_input, _ = tensor(args.root / HF, '0.linear_attn.in_proj_z.input', 'BF16', [1, 1, 2048])
    hf_output, _ = tensor(args.root / HF, '0.linear_attn.in_proj_z.output', 'BF16', [1, 1, 4096])
    for base in [CAUSAL / 'first-two-layers', TILED]:
        assert exact_words((args.root / base / '0005.f32').read_bytes()) == hf_input
    fused = exact_words((args.root / CAUSAL / 'first-two-layers/0006.f32').read_bytes())
    tiled = exact_words((args.root / TILED / '0006.f32').read_bytes())
    fused_diff, tiled_diff = differences(fused, hf_output), differences(tiled, hf_output)
    assert [d['index'] for d in fused_diff] == [753, 1497, 2650]
    assert [d['index'] for d in tiled_diff] == [3471, 4065]
    payloads.update({'input.bf16': hf_input, 'hf.bf16': hf_output,
                     'fused.bf16': fused, 'observed.bf16': tiled})
    manifest = dict(
        schema='ferrule.numeric-bf16-tiled.hello-l0-gate.v1', shape=[1, 4096, 2048],
        expected_source='pinned native HF in_proj_z capture; no generated oracle',
        production_output_is_diagnostic_only=True, sameinput=True,
        source_sha256={str(p): pin for p, pin in PINS.items()},
        payload_sha256=PAYLOAD_PINS, weight=event['weight'], scale=event['scale'],
        production_scratch_bytes=8 << 20, production_tile_rows=2047,
        replay_tile_rows=[64, 257, 1024, 2047],
        fused_vs_hf=fused_diff, production_tiled_vs_hf=tiled_diff,
        hf_gate_kernel='not established by existing gate capture; do not infer from QKV',
        reference_or_tolerance_changed=False,
    )
    payloads['manifest.json'] = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        (args.output / name).write_bytes(data)
    (args.output / 'SHA256SUMS').write_text(''.join(
        hashlib.sha256(data).hexdigest() + '  ' + name + '\n'
        for name, data in sorted(payloads.items())))
    print(json.dumps(dict(output=str(args.output), fixture_sha256=sha256(args.output / 'SHA256SUMS'),
                          sameinput=True, fused_vs_hf=fused_diff, production_tiled_vs_hf=tiled_diff)))


if __name__ == '__main__':
    main()
