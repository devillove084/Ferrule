"""Opt-in pinned Torch process; no production CUDA libraries. Originals immutable."""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture', type=Path)
    parser.add_argument('--execute-cuda', action='store_true')
    args = parser.parse_args()
    if not args.execute_cuda:
        parser.error('GPU owner grant and --execute-cuda required')
    sys.path.insert(0, str(Path.cwd() / 'scripts'))
    import qwen35_fp8_bf16_reference as ref
    pins = ref.initialize()
    import torch
    d = args.fixture
    manifest = json.loads((d / 'manifest.json').read_text())
    assert len(manifest['cases']) == 4 and not (d / 'oracle-manifest.json').exists(), 'use a fresh fixture, not a partial oracle rerun'
    for path, expected in manifest['pins'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    cases = list(manifest['cases'])
    # Fixed deterministic raw F32 operands; not CPU-computed expected arithmetic.
    for n in [1, 7, 32, 128, 256, 257, 1024]:
        for k in [1, 3, 8, 13, 32]:
            if k > n:
                continue
            for m in [1, 5]:
                name = f'dense-m{m}-n{n}-k{k}'
                raw = [((i * 37 + (i // n) * 17) % 131 - 65) * 0.03125 for i in range(m * n)]
                cases.append(dict(name=name, rows=m, experts=n, top_k=k, original=False, raw=raw))
    for name, raw, k in [
        ('ties', [1.] * 256, 8),
        ('raw-collision', [1e-30, 2e-30] + [0.] * 254, 8),
        ('underflow', [0.] + [-1000.] * 255, 32),
        ('wide-finite', [1000., 999., -1000.] + [0.] * 253, 8),
    ]:
        cases.append(dict(name=name, rows=1, experts=len(raw), top_k=k, original=False, raw=raw))
    report = []
    def write(name, stage, tensor):
        (d / f'{name}.{stage}.f32').write_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())
    with torch.inference_mode():
        for c in cases:
            name, m, n, k = c['name'], c['rows'], c['experts'], c['top_k']
            if not c['original']:
                raw = c.pop('raw')
                (d / (name + '.raw.f32')).write_bytes(struct.pack('<' + 'f' * len(raw), *raw))
            data = (d / (name + '.raw.f32')).read_bytes()
            raw_cpu = torch.frombuffer(bytearray(data), dtype=torch.float32).reshape(m, n)
            raw = raw_cpu.cuda()
            # Required contract ranks raw values; probability collisions are NOT
            # raw ties. For actuals these IDs must match original HF exactly.
            ids = torch.argsort(raw, dim=-1, descending=True, stable=True)[:, :k]
            id_bytes = ids.to(torch.int32).cpu().contiguous().numpy().tobytes()
            id_path = d / (name + '.ids.i32')
            if c['original']:
                assert id_path.read_bytes() == id_bytes, name
            else:
                id_path.write_bytes(id_bytes)
            probs = torch.softmax(raw, dim=-1, dtype=torch.float32)
            selected = probs.gather(-1, ids)
            denom = selected.sum(-1, keepdim=True)
            weights = selected / denom
            if c['original']:
                expected = (d / (name + '.weights.f32')).read_bytes()
                assert weights.cpu().contiguous().numpy().tobytes() == expected, (name, 'original reference changed')
            else:
                write(name, 'weights', weights)
            for stage, value in [('probabilities', probs), ('selected', selected), ('denominator', denom), ('normalized', weights)]:
                path = d / (name + '.' + stage + '.f32')
                if c.get('intermediate_reused'):
                    assert path.read_bytes() == value.cpu().contiguous().numpy().tobytes(), (name, stage)
                else:
                    write(name, stage, value)
            if name == 'raw-collision':
                assert raw[0, 0] != raw[0, 1] and probs[0, 0] == probs[0, 1]
                assert ids[0, :2].tolist() == [1, 0]
            report.append(dict(case=name, weights=m*k, original=c['original'], original_differences=0 if c['original'] else None))
    manifest['cases'] = cases
    manifest['oracle_pins'] = pins
    (d / 'oracle-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (d / 'oracle-cases.txt').write_text(''.join(f"{c['name']} {c['rows']} {c['experts']} {c['top_k']}\n" for c in cases))
    (d / 'oracle-report.json').write_text(json.dumps(report, indent=2) + '\n')
    generated = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(d.iterdir()) if p.is_file()}
    (d / 'GENERATED-SHA256.json').write_text(json.dumps(generated, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
