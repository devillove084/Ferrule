#!/usr/bin/env python3
"""CPU-only extraction of the frozen Hello L0 SAMEINPUT experiment, not an oracle exporter.

No Torch import, model execution, dequantization, or expected-value synthesis.
HF expected bytes come from the existing immutable profile reference and are
cross-checked against its independent native-HF producer capture.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

CAUSAL = Path('target/validation/bf16-full40-causal-20261004')
REFERENCE = Path('target/validation/numeric-bf16-gate-20261004-v2/profile-reference')
PINS = {
    CAUSAL / 'first-two-layers/events.jsonl': '3379c4be7e4d9c598acdf026d30004cdd8a353d841825a0cf5497614dc7dbdad',
    CAUSAL / 'first-two-layers/0003.f32': '04c24265212cc6828ef676790aa373c1202e3f8f9198874faffb7ce03435a4fe',
    CAUSAL / 'first-two-layers/0004.f32': 'e6ccd2a6d2c48c6f05c541087596e6a28174ffae79cef16928fdcd42402fb2a4',
    CAUSAL / 'hf-first-two-layers.safetensors': '41745fc2edaa2616e0018c7311144ea19bd6106fd5bb66835a7c0b800664d071',
    REFERENCE / 'manifest.json': '351495921233e0158eb2a4507c415d173c895a6086d5905b7a72d82de71849f9',
    REFERENCE / 'hello.safetensors': '224217500f282bd3366e5c7743d1aea68499877ca668e3bf344007fd83091274',
}
PAYLOAD_PINS = {
    'weight': '6d243d48276c45ce1577a3ed4e5fab313bf3a604edae63e61c069ee470bc7bf1',
    'scale': '6a3fd389a57abba3e96c4c522d78e6af64dcc76002d26557e0d50fc43603df58',
}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def header(path):
    with path.open('rb') as f:
        length, = struct.unpack('<Q', f.read(8))
        assert length <= 16 * 1024 * 1024
        return json.loads(f.read(length)), length + 8


def tensor(path, name, dtype, shape):
    entries, base = header(path)
    entry = entries[name]
    assert entry['dtype'] == dtype and entry['shape'] == shape, entry
    begin, end = entry['data_offsets']
    assert 0 <= begin < end and end - begin <= 16 * 1024 * 1024
    with path.open('rb') as f:
        f.seek(base + begin)
        result = f.read(end - begin)
    assert len(result) == end - begin
    return result, base + begin


def exact_words(data):
    words = []
    for (word,) in struct.iter_unpack('<I', data):
        assert word & 0xffff == 0, 'capture is not already exactly BF16'
        assert word & 0x7f800000 != 0x7f800000, 'nonfinite capture'
        words.append(word >> 16)
    return struct.pack('<' + 'H' * len(words), *words)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for relative, pin in PINS.items():
        assert sha256(args.root / relative) == pin, relative
    events = [json.loads(s) for s in (args.root / CAUSAL / 'first-two-layers/events.jsonl').read_text().splitlines()]
    event = events[3]
    assert event['event'] == 'linear.input' and event['layer'] == 0
    assert event['shape'] == [1, 2048]
    assert event['parameter'] == 'layers.0.linear_attention.query_key_value.weight'
    payloads = {}
    for kind, dtype, shape, filename in [
        ('weight', 'F8_E4M3', [8192, 2048], 'weight.fp8'),
        ('scale', 'BF16', [64, 16], 'scale.bf16'),
    ]:
        source = event[kind]
        data, offset = tensor(Path(source['path']), source['name'], dtype, shape)
        assert offset == source['offset'] and len(data) == source['bytes']
        assert hashlib.sha256(data).hexdigest() == PAYLOAD_PINS[kind]
        payloads[filename] = data
    input_words = exact_words((args.root / CAUSAL / 'first-two-layers/0003.f32').read_bytes())
    observed_words = exact_words((args.root / CAUSAL / 'first-two-layers/0004.f32').read_bytes())
    hf_path = args.root / CAUSAL / 'hf-first-two-layers.safetensors'
    hf_input, _ = tensor(hf_path, '0.linear_attn.in_proj_qkv.input', 'BF16', [1, 1, 2048])
    hf_output, _ = tensor(hf_path, '0.linear_attn.in_proj_qkv.output', 'BF16', [1, 1, 8192])
    history, _ = tensor(args.root / REFERENCE / 'hello.safetensors', 'prefill.cache.conv_states.00', 'BF16', [1, 8192, 4])
    reference = b''.join(history[i + 6:i + 8] for i in range(0, len(history), 8))
    assert input_words == hf_input, 'not SAMEINPUT'
    assert hf_output == reference, 'native producer differs from immutable reference'
    unequal = sum(observed_words[i:i + 2] != reference[i:i + 2] for i in range(0, len(reference), 2))
    assert unequal == 11, unequal
    payloads.update({'input.bf16': input_words, 'observed.bf16': observed_words, 'hf.bf16': reference})
    manifest = dict(
        schema='ferrule.numeric-bf16-tiled.hello-l0-qkv.v1', shape=[1, 8192, 2048],
        expected_source='pinned native HF output, independently equals profile conv last token',
        production_output_is_diagnostic_only=True, observed_unequal=unequal,
        source_sha256={str(p): pin for p, pin in PINS.items()},
        payload_sha256=PAYLOAD_PINS, weight=event['weight'], scale=event['scale'],
        candidate='explicit Bf16RneF32Accumulate plan through typed BF16 API; not a model profile',
        layout='A row-major; W rows become B column-major; CTA64x64x32 warp32x32x32 MMA16x8x16; 3 stages',
        hf_kernel='cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x1_tn_align8',
        hf_source_sha256='0f8ca77e875c31d448137c9ac291bed505c6b9927a85b5b8460af29d1e1ee647',
        torch='2.13.0+cu129', transformers='5.5.3',
        allow_tf32=False, allow_bf16_reduced_precision_reduction=False,
    )
    payloads['manifest.json'] = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        (args.output / name).write_bytes(data)
    (args.output / 'SHA256SUMS').write_text(''.join(
        hashlib.sha256(data).hexdigest() + '  ' + name + '\n'
        for name, data in sorted(payloads.items())))
    print(json.dumps(dict(output=str(args.output), fixture_sha256=sha256(args.output / 'SHA256SUMS'),
                          elements=8192, hf_vs_observed=unequal, sameinput=True)))


if __name__ == '__main__':
    main()
