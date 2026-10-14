"""CPU-only SAMEINPUT extraction. Never imports Torch or generates HF values."""
import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source / 'manifest.json').read_text())
    cases, pins = [], {}
    for case in manifest['cases']:
        if case['module'] != 'mlp.gate':
            continue
        name, m, n = case['id'], case['m'], case['n']
        hf, direct = source / (name + '.hf.bf16'), source / (name + '.direct.bf16')
        assert direct.read_bytes() == hf.read_bytes(), (name, 'projection not SAMEINPUT')
        raw = b''.join(struct.pack('<I', word[0] << 16) for word in struct.iter_unpack('<H', hf.read_bytes()))
        assert len(raw) == 4 * m * n
        (destination / (name + '.raw.f32')).write_bytes(raw)
        ids = [v[0] for v in struct.iter_unpack('<q', (source / (name + '.hf.ids')).read_bytes())]
        values = [v[0] for v in struct.iter_unpack('<f', raw)]
        for row in range(m):
            expected = sorted(range(n), key=lambda i: (-values[row * n + i], i))[:8]
            assert expected == ids[row * 8:(row + 1) * 8], (name, row, 'raw ranking differs')
        (destination / (name + '.ids.i32')).write_bytes(struct.pack('<' + 'i' * len(ids), *ids))
        shutil.copyfile(source / (name + '.hf.weights'), destination / (name + '.weights.f32'))
        # Only reuse intermediate evidence with identical raw input. Old hello3
        # came from the pre-projection-fix logits; it is explicitly excluded.
        reused = (source / (name + '.actual.bf16')).read_bytes() == hf.read_bytes()
        if reused:
            for stage in ['probabilities', 'selected', 'denominator', 'normalized']:
                shutil.copyfile(source / (name + '.' + stage + '.f32'), destination / (name + '.' + stage + '.f32'))
            assert (destination / (name + '.normalized.f32')).read_bytes() == (destination / (name + '.weights.f32')).read_bytes()
        for suffix in ['hf.bf16', 'direct.bf16', 'hf.ids', 'hf.weights', 'actual.bf16']:
            p = source / (name + '.' + suffix)
            pins[str(p.resolve())] = digest(p)
        cases.append(dict(name=name, rows=m, experts=n, top_k=8, original=True,
                          raw_bit_differences=0, intermediate_reused=reused))
    assert len(cases) == 4
    (destination / 'cases.txt').write_text(''.join(f"{c['name']} {c['rows']} {c['experts']} {c['top_k']}\n" for c in cases))
    (destination / 'manifest.json').write_text(json.dumps(dict(cases=cases, pins=pins), indent=2) + '\n')
    return cases


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination), indent=2))
