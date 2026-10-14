"""CPU-only all-stage bitwise report. Failure is durable and exits nonzero."""
import argparse
import json
import struct
from pathlib import Path


def floats(data):
    return [x[0] for x in struct.iter_unpack('<f', data)]


def compare(a, b):
    assert len(a) == len(b)
    bad = [i for i in range(len(a) // 4) if a[4*i:4*i+4] != b[4*i:4*i+4]]
    av, bv = floats(a), floats(b)
    return dict(elements=len(av), unequal=len(bad), first=bad[0] if bad else None,
                max_abs=max((abs(x-y) for x, y in zip(av, bv)), default=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fixture', type=Path)
    p.add_argument('--require-production', action='store_true')
    args = p.parse_args()
    d = args.fixture
    report = []
    cases = d / ('oracle-cases.txt' if (d / 'oracle-cases.txt').exists() else 'cases.txt')
    for line in cases.read_text().splitlines():
        name, m, n, k = line.split()
        m, n, k = int(m), int(n), int(k)
        get = lambda suffix: (d / (name + '.' + suffix)).read_bytes()
        trace = get('trace.f32')
        stride = (2*n+k+3)*4
        assert len(trace) == m * stride
        def stage(offset, width):
            return b''.join(trace[row*stride+4*offset:row*stride+4*(offset+width)] for row in range(m))
        stages = dict(raw=compare(get('probe.raw.f32'), get('raw.f32')))
        for label, actual in [('probabilities', stage(2+n, n)), ('selected', stage(2+2*n, k)), ('denominator', stage(2+2*n+k, 1))]:
            stages[label] = compare(actual, get(label + '.f32'))
        stages['weights'] = compare(get('probe.weights.f32'), get('weights.f32'))
        ids = get('probe.ids.i32')
        expected_ids = get('ids.i32')
        assert len(ids) == len(expected_ids)
        stages['ids'] = dict(elements=m*k, unequal=sum(ids[i:i+4] != expected_ids[i:i+4] for i in range(0, len(ids), 4)))
        if args.require_production:
            stages['production_vs_trace'] = compare(get('production.f32'), get('probe.weights.f32'))
        report.append(dict(case=name, stages=stages, first_stage=next((key for key,value in stages.items() if value['unequal']), None)))
    result = dict(cases=report, passing=sum(c['first_stage'] is None for c in report), total=len(report),
                  note='Max/exp/full-sum trace retained diagnostically; independent HF exposes probabilities, selected, selected-sum and weights. No tolerance.')
    (d / ('stage-report-production.json' if args.require_production else 'stage-report.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return int(result['passing'] != result['total'])


if __name__ == '__main__':
    raise SystemExit(main())
