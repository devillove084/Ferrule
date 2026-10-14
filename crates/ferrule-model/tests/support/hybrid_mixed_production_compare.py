"""CPU-only bounded boundary diagnosis; never an all-inventory acceptance report.

The unmodified full comparison requires 2178 tensors/case. This explicitly
reports uncovered tensors and compares only captured Hello/layer0 boundaries.
Expected values are read from the pinned independent archive, never Ferrule.
"""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

PIN = 'fb96ba4dd134761af30b4025b4cce86f96136cb01f05b5b4fca3e74587caa71d'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class Archive:
    def __init__(self, path):
        self.path = path
        with path.open('rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            assert 0 < n <= 64 << 20
            self.header = json.loads(f.read(n))
            self.offset = n + 8

    def values(self, name):
        entry = self.header[name]
        start, end = entry['data_offsets']
        assert 0 <= start <= end <= self.path.stat().st_size - self.offset
        with self.path.open('rb') as f:
            f.seek(self.offset + start)
            data = f.read(end - start)
        dtype = entry['dtype']
        if dtype == 'BF16':
            values = [struct.unpack('<f', struct.pack('<I', word << 16))[0]
                      for (word,) in struct.iter_unpack('<H', data)]
        else:
            assert dtype in ('F32', 'I64')
            values = [v[0] for v in struct.iter_unpack('<f' if dtype == 'F32' else '<q', data)]
        assert len(values) == math.prod(entry['shape'])
        return values


def metrics(actual, expected, ids=False):
    assert len(actual) == len(expected) and actual
    unequal, failures, square, maximum, first = 0, 0, 0., 0., []
    for index, (a, e) in enumerate(zip(actual, expected)):
        assert math.isfinite(a) and math.isfinite(e)
        exact = a == e if ids else struct.pack('<f', a) == struct.pack('<f', e)
        d = abs(a - e)
        unequal += not exact
        failures += (not exact) if ids else d > .00025 + .012 * abs(e)
        square += d * d
        maximum = max(maximum, d)
        if not exact and len(first) < 8:
            first.append(dict(index=index, actual=a, expected=e,
                              actual_bits=None if ids else struct.unpack('<I', struct.pack('<f', a))[0],
                              expected_bits=None if ids else struct.unpack('<I', struct.pack('<f', e))[0]))
    return dict(compared=len(actual), exact=len(actual)-unequal, unequal=unequal,
                failures=failures, max_abs=maximum, rmse=math.sqrt(square/len(actual)), first=first)


def mapping(event, rank):
    p = f'prefill.rank.{rank}.layers.00.'
    key = event['event']
    names = {
        'embedding.output': 'prefill.embedding', 'mixed.layer.input': 'prefill.embedding',
        'gdn.conv_state': f'prefill.rank.{rank}.cache.conv_states.00',
        'gdn.recurrent_state': f'prefill.rank.{rank}.cache.recurrent_states.00',
        'mixed.attention.partial': p+'attention_partial',
        'mixed.attention.reduced': 'prefill.layers.00.attention',
        'mixed.routed.partial': p+'routed_partial',
        'mixed.moe.partial': p+'moe_partial',
        'mixed.moe.reduced': 'prefill.layers.00.moe',
        'mixed.layer.output': 'prefill.layers.00.hidden',
        'router': p+'raw_router_logits',
    }
    if key == 'mixed.swiglu.partial' and event['role'] == 'SharedExpertDown':
        return p+'shared_partial'
    if key == 'mixed.linear.output' and event['role'] == 'SharedExpertOutputGate':
        return p+'shared_gate_logit'
    return names.get(key)


def compare(reference, capture):
    assert sha(reference/'manifest.json') == PIN
    manifest = json.loads((reference/'manifest.json').read_text())
    case = next(c for c in manifest['cases'] if c['id'] == 'hello')
    path = reference/case['file']
    assert sha(path) == case['sha256']
    archive = Archive(path)
    inventory = {k for k in archive.header if k != '__metadata__' and not k.endswith('.native_expert_ids')}
    assert len(inventory) == 2178
    rows, diagnostic_only, covered = [], [], set()
    for rank in range(2):
        directory = capture/f'rank-{rank}'
        events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
        assert events and events[-1]['event'] == 'mixed.layer.output'
        assert all(e['layer'] in (None, 0) for e in events)
        for index, event in enumerate(events):
            data = (directory/event['file']).read_bytes()
            actual = [v[0] for v in struct.iter_unpack('<f', data)]
            assert len(actual) == math.prod(event['shape'])
            key = mapping(event, rank)
            if key:
                rows.append(dict(rank=rank, order=index, event=event['event'], reference=key,
                                 **metrics(actual, archive.values(key))))
                covered.add(key)
            else:
                diagnostic_only.append(dict(rank=rank, order=index, **event))
            if event['event'] == 'router':
                for field, suffix in [('ids','selected_expert_ids'),('weights','routing_weights')]:
                    key = f'prefill.rank.{rank}.layers.00.{suffix}'
                    rows.append(dict(rank=rank,order=index,event=field,reference=key,
                                     **metrics(event[field],archive.values(key),field=='ids')))
                    covered.add(key)
    return dict(schema='ferrule.mixed-layer0-diagnosis.v1', acceptance=False,
                reference_manifest_sha256=PIN, expected_candidate_tensors=2178,
                distinct_compared_tensors=len(covered), missing_count=len(inventory-covered),
                first_inexact_by_rank={str(r):next((v for v in rows if v['rank']==r and v['unequal']),None) for r in range(2)},
                comparisons=rows, diagnostic_only=diagnostic_only,
                atol=.00025, rtol=.012,
                note='QKV/core/norm/residual events retained as diagnostics only where the immutable oracle lacks that intermediate; no synthesized expected tensors; final head/later layers not executed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--capture', required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.reference, args.capture)
    with (args.capture/'boundary-comparison.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('comparisons','diagnostic_only')}, indent=2))
    for row in result['comparisons']:
        print(json.dumps({k:v for k,v in row.items() if k != 'first'}))
